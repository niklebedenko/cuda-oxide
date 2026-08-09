//! Stable, target-directory-independent identity for one collected device module.
//!
//! Cargo and rustc crate hashes intentionally cover more than device code. In
//! particular, they may include build-script outputs and dependency metadata
//! which differ between otherwise equivalent target directories. They are a
//! sound incremental-compilation boundary, but too coarse to key a shared
//! device artifact cache.
//!
//! This module instead hashes the exact rustc inputs consumed by the device
//! importer. It starts from the collected monomorphizations, then closes over
//! concrete types, layouts, enum discriminants, and constant/static allocation
//! data. The implementation deliberately fails closed: an input which cannot
//! be normalized or a global-allocation kind which the importer cannot consume
//! makes the cache unavailable for that owner.

use crate::collector::{CollectionResult, DeviceExternAttrs};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_index::Idx;
use rustc_middle::ich::StableHashingContext;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::mir::interpret::{AllocId, ConstAllocation, GlobalAlloc, Scalar};
use rustc_middle::mir::visit::Visitor as MirVisitor;
use rustc_middle::mir::{Body, ConstOperand, ConstValue};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{
    EarlyBinder, FnSig, Instance, Ty, TyCtxt, TyKind, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt, TypeVisitor, TypingEnv,
};
use rustc_target::callconv::FnAbi;
use std::fmt;
use std::ops::ControlFlow;

/// Bump whenever the semantic-resource closure or framing changes.
const SEMANTIC_FINGERPRINT_VERSION: &str = "cuda-oxide-device-semantic-v3";

/// A rustc-stable 128-bit digest of all inputs which can affect one device module.
///
/// The surrounding artifact-cache key hashes these bytes together with the
/// backend, finalizer, target, and CUDA Oxide option identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceSemanticFingerprint([u8; 16]);

impl DeviceSemanticFingerprint {
    pub(crate) fn to_hex(self) -> String {
        use std::fmt::Write as _;

        let mut output = String::with_capacity(32);
        for byte in self.0 {
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
        }
        output
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeviceSemanticFingerprintError {
    context: String,
}

impl DeviceSemanticFingerprintError {
    fn new(context: impl Into<String>) -> Self {
        Self {
            context: context.into(),
        }
    }
}

impl fmt::Display for DeviceSemanticFingerprintError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.context)
    }
}

impl std::error::Error for DeviceSemanticFingerprintError {}

/// Compute a stable semantic digest for the exact collected device module.
///
/// This function is intentionally called after device collection but before
/// MIR import/codegen. Its work is bounded by the already-collected function
/// graph and the type/static resources those functions reference.
pub(crate) fn device_semantic_fingerprint<'tcx>(
    tcx: TyCtxt<'tcx>,
    collection: &CollectionResult<'tcx>,
) -> Result<DeviceSemanticFingerprint, DeviceSemanticFingerprintError> {
    let resources = SemanticResources::collect(tcx, collection)?;
    let fingerprint: Fingerprint = tcx.with_stable_hashing_context(|mut hcx| {
        let mut hasher = StableHasher::new();
        hash_resources(&resources, &mut hcx, &mut hasher);
        hasher.finish()
    });
    Ok(DeviceSemanticFingerprint(fingerprint.to_le_bytes()))
}

struct FunctionResource<'tcx> {
    symbol: String,
    export_name: String,
    is_kernel: bool,
    root_descriptor: Option<String>,
    instance: Instance<'tcx>,
    call_abi: &'tcx FnAbi<'tcx, Ty<'tcx>>,
    attrs: &'tcx CodegenFnAttrs,
    body: &'tcx Body<'tcx>,
    evaluated_constants: Vec<ConstValue>,
    mono_successors: Vec<Vec<usize>>,
    codegen_policy: crate::device_codegen::DeviceFunctionCodegenPolicy,
    debug_source_scopes: Option<llvm_export::ops::DebugSourceScopeMap>,
}

struct TypeResource<'tcx> {
    ty: Ty<'tcx>,
    layout: TyAndLayout<'tcx>,
    discriminants: Vec<(u32, u128, Ty<'tcx>)>,
}

struct StaticResource<'tcx> {
    def_id: DefId,
    def_kind: DefKind,
    ty: Option<Ty<'tcx>>,
    attrs: &'tcx CodegenFnAttrs,
    initializer: ConstAllocation<'tcx>,
}

struct AllocationResource<'tcx> {
    allocation: GlobalAlloc<'tcx>,
}

struct DeviceExternResource<'tcx> {
    stable_crate_id: u64,
    export_name: String,
    signature: FnSig<'tcx>,
    attrs: DeviceExternAttrs,
}

struct FamilyResource {
    root_symbol: String,
    root_export_name: String,
    function_symbols: Vec<String>,
}

struct SemanticResources<'tcx> {
    functions: Vec<FunctionResource<'tcx>>,
    types: Vec<TypeResource<'tcx>>,
    allocations: Vec<AllocationResource<'tcx>>,
    statics: Vec<StaticResource<'tcx>>,
    device_externs: Vec<DeviceExternResource<'tcx>>,
    families: Vec<FamilyResource>,
    core_index_trait: Option<DefId>,
}

impl<'tcx> SemanticResources<'tcx> {
    fn collect(
        tcx: TyCtxt<'tcx>,
        collection: &CollectionResult<'tcx>,
    ) -> Result<Self, DeviceSemanticFingerprintError> {
        let mut collector = ResourceCollector::new(tcx);

        let policies =
            crate::device_codegen::device_function_codegen_policies(tcx, &collection.functions);
        let mut functions = collection
            .functions
            .iter()
            .zip(policies)
            .collect::<Vec<_>>();
        functions.sort_by_cached_key(|(function, _)| {
            (
                tcx.symbol_name(function.instance).name.to_string(),
                function.export_name.clone(),
            )
        });
        for (function, policy) in functions {
            collector.add_function(function, policy)?;
        }

        let mut device_externs = collection.device_externs.iter().collect::<Vec<_>>();
        device_externs.sort_by_cached_key(|declaration| {
            (
                tcx.stable_crate_id(declaration.def_id.krate).as_u64(),
                declaration.export_name.clone(),
            )
        });
        for declaration in device_externs {
            collector.add_device_extern(declaration)?;
        }

        collector.close_type_graph()?;
        collector.close_allocation_graph()?;
        // Static types and initializers can add further type/allocation roots.
        // Iterate to a fixed point without relying on hash-table iteration.
        while collector.has_unclosed_resources() {
            collector.close_type_graph()?;
            collector.close_allocation_graph()?;
        }

        let mut families = collection
            .function_families
            .iter()
            .map(|family| FamilyResource {
                root_symbol: family.root_symbol.clone(),
                root_export_name: family.root_export_name.clone(),
                function_symbols: family.function_symbols.clone(),
            })
            .collect::<Vec<_>>();
        families.sort_by(|left, right| {
            (&left.root_symbol, &left.root_export_name)
                .cmp(&(&right.root_symbol, &right.root_export_name))
        });

        Ok(Self {
            functions: collector.functions,
            types: collector.types,
            allocations: collector.allocations,
            statics: collector.statics,
            device_externs: collector.device_externs,
            families,
            core_index_trait: tcx.lang_items().index_trait(),
        })
    }
}

struct ResourceCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    functions: Vec<FunctionResource<'tcx>>,
    types: Vec<TypeResource<'tcx>>,
    allocations: Vec<AllocationResource<'tcx>>,
    statics: Vec<StaticResource<'tcx>>,
    device_externs: Vec<DeviceExternResource<'tcx>>,
    type_queue: Vec<Ty<'tcx>>,
    type_cursor: usize,
    seen_types: FxHashSet<Ty<'tcx>>,
    allocation_queue: Vec<AllocId>,
    allocation_cursor: usize,
    seen_allocations: FxHashSet<AllocId>,
    seen_statics: FxHashSet<DefId>,
}

impl<'tcx> ResourceCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            functions: Vec::new(),
            types: Vec::new(),
            allocations: Vec::new(),
            statics: Vec::new(),
            device_externs: Vec::new(),
            type_queue: Vec::new(),
            type_cursor: 0,
            seen_types: FxHashSet::default(),
            allocation_queue: Vec::new(),
            allocation_cursor: 0,
            seen_allocations: FxHashSet::default(),
            seen_statics: FxHashSet::default(),
        }
    }

    fn add_function(
        &mut self,
        function: &crate::collector::CollectedFunction<'tcx>,
        codegen_policy: crate::device_codegen::DeviceFunctionCodegenPolicy,
    ) -> Result<(), DeviceSemanticFingerprintError> {
        let instance = function.instance;
        let def_id = instance.def_id();
        let symbol = self.tcx.symbol_name(instance).name.to_string();
        let call_abi = self
            .tcx
            .fn_abi_of_instance(
                TypingEnv::fully_monomorphized()
                    .as_query_input((instance, rustc_middle::ty::List::empty())),
            )
            .map_err(|error| {
                DeviceSemanticFingerprintError::new(format!(
                    "cannot compute device call ABI for `{symbol}`: {error:?}"
                ))
            })?;
        let body = self.tcx.instance_mir(instance.def);
        let debug_source_scopes =
            crate::device_codegen::device_debug_kind(self.tcx.sess.opts.debuginfo)
                .line_tables_enabled()
                .then(|| crate::device_codegen::device_debug_source_scope_map(self.tcx, function));

        self.seed_type(instance.ty(self.tcx, TypingEnv::fully_monomorphized()))?;
        self.seed_type(call_abi.ret.layout.ty)?;
        for argument in &call_abi.args {
            self.seed_type(argument.layout.ty)?;
        }
        let evaluated_constants = self.seed_body_resources(instance, body, &symbol)?;

        let reachability = crate::collector::device_mono_reachability(self.tcx, instance);
        self.functions.push(FunctionResource {
            symbol,
            export_name: function.export_name.clone(),
            is_kernel: function.is_kernel,
            root_descriptor: function.root_descriptor.clone(),
            instance,
            call_abi,
            attrs: self.tcx.codegen_fn_attrs(def_id),
            body,
            evaluated_constants,
            mono_successors: reachability.successors,
            codegen_policy,
            debug_source_scopes,
        });
        Ok(())
    }

    fn add_device_extern(
        &mut self,
        declaration: &crate::collector::DeviceExternDecl,
    ) -> Result<(), DeviceSemanticFingerprintError> {
        let signature = self.tcx.instantiate_bound_regions_with_erased(
            self.tcx.fn_sig(declaration.def_id).instantiate_identity(),
        );
        self.seed_signature_types(signature)?;
        self.device_externs.push(DeviceExternResource {
            stable_crate_id: self.tcx.stable_crate_id(declaration.def_id.krate).as_u64(),
            export_name: declaration.export_name.clone(),
            signature,
            attrs: declaration.attrs.clone(),
        });
        Ok(())
    }

    fn seed_signature_types(
        &mut self,
        signature: FnSig<'tcx>,
    ) -> Result<(), DeviceSemanticFingerprintError> {
        for ty in signature.inputs_and_output.iter() {
            self.seed_type(ty)?;
        }
        Ok(())
    }

    fn seed_body_resources(
        &mut self,
        instance: Instance<'tcx>,
        body: &'tcx Body<'tcx>,
        symbol: &str,
    ) -> Result<Vec<ConstValue>, DeviceSemanticFingerprintError> {
        let mut type_seeds = InstantiatedTypeSeeds {
            tcx: self.tcx,
            instance,
            types: Vec::new(),
        };
        if let ControlFlow::Break(error) = body.visit_with(&mut type_seeds) {
            return Err(DeviceSemanticFingerprintError::new(format!(
                "cannot normalize a MIR type in device function `{symbol}`: {error}"
            )));
        }
        for ty in type_seeds.types {
            self.seed_type(ty)?;
        }

        let mut constant_seeds = ConstantAllocationSeeds {
            tcx: self.tcx,
            instance,
            allocation_ids: Vec::new(),
            values: Vec::new(),
            error: None,
        };
        constant_seeds.visit_body(body);
        if let Some(error) = constant_seeds.error {
            return Err(DeviceSemanticFingerprintError::new(format!(
                "cannot fingerprint a MIR constant in device function `{symbol}`: {error}"
            )));
        }
        for allocation_id in constant_seeds.allocation_ids {
            self.seed_allocation(allocation_id);
        }
        Ok(constant_seeds.values)
    }

    fn seed_type(&mut self, ty: Ty<'tcx>) -> Result<(), DeviceSemanticFingerprintError> {
        let ty = self
            .tcx
            .try_normalize_erasing_regions(TypingEnv::fully_monomorphized(), ty)
            .map_err(|_| {
                DeviceSemanticFingerprintError::new(format!(
                    "cannot normalize concrete device type `{ty}`"
                ))
            })?;
        let ty = self.tcx.erase_and_anonymize_regions(ty);
        if ty.has_non_region_param() {
            return Err(DeviceSemanticFingerprintError::new(format!(
                "device semantic type remains generic after normalization: `{ty}`"
            )));
        }
        if self.seen_types.insert(ty) {
            self.type_queue.push(ty);
        }
        Ok(())
    }

    fn close_type_graph(&mut self) -> Result<(), DeviceSemanticFingerprintError> {
        while self.type_cursor < self.type_queue.len() {
            let ty = self.type_queue[self.type_cursor];
            self.type_cursor += 1;

            let layout = self
                .tcx
                .layout_of(TypingEnv::fully_monomorphized().as_query_input(ty))
                .map_err(|error| {
                    DeviceSemanticFingerprintError::new(format!(
                        "cannot compute device semantic layout for `{ty}`: {error:?}"
                    ))
                })?;

            let mut discriminants = Vec::new();
            if let TyKind::Adt(definition, arguments) = ty.kind() {
                if definition.is_enum() {
                    discriminants.extend(definition.discriminants(self.tcx).map(
                        |(variant, discriminant)| {
                            (variant.index() as u32, discriminant.val, discriminant.ty)
                        },
                    ));
                }
                for variant in definition.variants() {
                    for field in &variant.fields {
                        let field_ty = field.ty(self.tcx, arguments);
                        self.seed_type(field_ty)?;
                    }
                }
            }

            let mut nested = NestedTypeSeeds { types: Vec::new() };
            if let ControlFlow::Break(()) = ty.super_visit_with(&mut nested) {
                unreachable!("nested type collection never stops early");
            }
            for nested_ty in nested.types {
                self.seed_type(nested_ty)?;
            }

            self.types.push(TypeResource {
                ty,
                layout,
                discriminants,
            });
        }
        Ok(())
    }

    fn seed_allocation(&mut self, allocation_id: AllocId) {
        if self.seen_allocations.insert(allocation_id) {
            self.allocation_queue.push(allocation_id);
        }
    }

    fn close_allocation_graph(&mut self) -> Result<(), DeviceSemanticFingerprintError> {
        while self.allocation_cursor < self.allocation_queue.len() {
            let allocation_id = self.allocation_queue[self.allocation_cursor];
            self.allocation_cursor += 1;
            let allocation = self.tcx.global_alloc(allocation_id);

            match &allocation {
                GlobalAlloc::Memory(memory) => {
                    for provenance in memory.inner().provenance().ptrs().values() {
                        self.seed_allocation(provenance.alloc_id());
                    }
                }
                GlobalAlloc::Static(def_id) => self.add_static(*def_id)?,
                GlobalAlloc::Function { .. }
                | GlobalAlloc::VTable(..)
                | GlobalAlloc::TypeId { .. } => {
                    return Err(DeviceSemanticFingerprintError::new(format!(
                        "device constant references unsupported global allocation kind: {allocation:?}"
                    )));
                }
            }
            self.allocations.push(AllocationResource { allocation });
        }
        Ok(())
    }

    fn add_static(&mut self, def_id: DefId) -> Result<(), DeviceSemanticFingerprintError> {
        if !self.seen_statics.insert(def_id) {
            return Ok(());
        }

        let def_kind = self.tcx.def_kind(def_id);
        let DefKind::Static { nested, .. } = def_kind else {
            return Err(DeviceSemanticFingerprintError::new(format!(
                "global allocation `{}` is not a static",
                self.tcx.def_path_str(def_id)
            )));
        };
        let ty = if nested {
            None
        } else {
            let ty = self.tcx.type_of(def_id).no_bound_vars().ok_or_else(|| {
                DeviceSemanticFingerprintError::new(format!(
                    "device static `{}` has generic type parameters",
                    self.tcx.def_path_str(def_id)
                ))
            })?;
            self.seed_type(ty)?;
            Some(ty)
        };
        let initializer = self.tcx.eval_static_initializer(def_id).map_err(|error| {
            DeviceSemanticFingerprintError::new(format!(
                "cannot evaluate device static `{}`: {error:?}",
                self.tcx.def_path_str(def_id)
            ))
        })?;
        for provenance in initializer.inner().provenance().ptrs().values() {
            self.seed_allocation(provenance.alloc_id());
        }
        self.statics.push(StaticResource {
            def_id,
            def_kind,
            ty,
            attrs: self.tcx.codegen_fn_attrs(def_id),
            initializer,
        });
        Ok(())
    }

    fn has_unclosed_resources(&self) -> bool {
        self.type_cursor < self.type_queue.len()
            || self.allocation_cursor < self.allocation_queue.len()
    }
}

struct InstantiatedTypeSeeds<'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    types: Vec<Ty<'tcx>>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for InstantiatedTypeSeeds<'tcx> {
    type Result = ControlFlow<String>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        match self
            .instance
            .try_instantiate_mir_and_normalize_erasing_regions(
                self.tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(ty),
            ) {
            Ok(ty) => {
                self.types.push(ty);
                ControlFlow::Continue(())
            }
            Err(error) => ControlFlow::Break(format!("{error:?}")),
        }
    }
}

struct NestedTypeSeeds<'tcx> {
    types: Vec<Ty<'tcx>>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for NestedTypeSeeds<'tcx> {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        self.types.push(ty);
        ty.super_visit_with(self)
    }
}

struct ConstantAllocationSeeds<'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    allocation_ids: Vec<AllocId>,
    values: Vec<ConstValue>,
    error: Option<String>,
}

impl<'tcx> ConstantAllocationSeeds<'tcx> {
    fn record_value(&mut self, value: ConstValue) {
        self.values.push(value);
        let allocation_id = match value {
            ConstValue::Scalar(Scalar::Ptr(pointer, _)) => Some(pointer.provenance.alloc_id()),
            ConstValue::Indirect { alloc_id, .. } | ConstValue::Slice { alloc_id, .. } => {
                Some(alloc_id)
            }
            ConstValue::Scalar(Scalar::Int(..)) | ConstValue::ZeroSized => None,
        };
        if let Some(allocation_id) = allocation_id {
            self.allocation_ids.push(allocation_id);
        }
    }
}

impl<'tcx> MirVisitor<'tcx> for ConstantAllocationSeeds<'tcx> {
    fn visit_const_operand(
        &mut self,
        constant: &ConstOperand<'tcx>,
        location: rustc_middle::mir::Location,
    ) {
        if self.error.is_some() {
            return;
        }
        let instantiated = self
            .instance
            .try_instantiate_mir_and_normalize_erasing_regions(
                self.tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(*constant),
            );
        match instantiated {
            Ok(constant) => match constant.const_.eval(
                self.tcx,
                TypingEnv::fully_monomorphized(),
                constant.span,
            ) {
                Ok(value) => self.record_value(value),
                Err(error) => self.error = Some(format!("{error:?}")),
            },
            Err(error) => self.error = Some(format!("{error:?}")),
        }
        self.super_const_operand(constant, location);
    }
}

fn hash_resources(
    resources: &SemanticResources<'_>,
    hcx: &mut StableHashingContext<'_>,
    hasher: &mut StableHasher,
) {
    SEMANTIC_FINGERPRINT_VERSION.hash_stable(hcx, hasher);

    "functions".hash_stable(hcx, hasher);
    resources.functions.len().hash_stable(hcx, hasher);
    for function in &resources.functions {
        function.symbol.hash_stable(hcx, hasher);
        function.export_name.hash_stable(hcx, hasher);
        function.is_kernel.hash_stable(hcx, hasher);
        function.root_descriptor.hash_stable(hcx, hasher);
        function.instance.hash_stable(hcx, hasher);
        function.call_abi.hash_stable(hcx, hasher);
        function.attrs.hash_stable(hcx, hasher);
        function.body.hash_stable(hcx, hasher);
        function.evaluated_constants.hash_stable(hcx, hasher);
        hash_successors(&function.mono_successors, hcx, hasher);
        inline_attr_tag(function.codegen_policy.inline_attr).hash_stable(hcx, hasher);
        function
            .codegen_policy
            .device_link_always
            .hash_stable(hcx, hasher);
        function
            .codegen_policy
            .device_link_inline_candidate
            .hash_stable(hcx, hasher);
        function
            .codegen_policy
            .deferred_full_unroll
            .hash_stable(hcx, hasher);
        hash_debug_source_scope_map(function.debug_source_scopes.as_ref(), hcx, hasher);
    }

    "types".hash_stable(hcx, hasher);
    resources.types.len().hash_stable(hcx, hasher);
    for resource in &resources.types {
        resource.ty.hash_stable(hcx, hasher);
        resource.layout.hash_stable(hcx, hasher);
        resource.discriminants.len().hash_stable(hcx, hasher);
        for (variant, value, ty) in &resource.discriminants {
            variant.hash_stable(hcx, hasher);
            value.hash_stable(hcx, hasher);
            ty.hash_stable(hcx, hasher);
        }
    }

    "allocations".hash_stable(hcx, hasher);
    resources.allocations.len().hash_stable(hcx, hasher);
    for resource in &resources.allocations {
        resource.allocation.hash_stable(hcx, hasher);
    }

    "statics".hash_stable(hcx, hasher);
    resources.statics.len().hash_stable(hcx, hasher);
    for resource in &resources.statics {
        resource.def_id.hash_stable(hcx, hasher);
        resource.def_kind.hash_stable(hcx, hasher);
        resource.ty.hash_stable(hcx, hasher);
        resource.attrs.hash_stable(hcx, hasher);
        resource.initializer.hash_stable(hcx, hasher);
    }

    "device-externs".hash_stable(hcx, hasher);
    resources.device_externs.len().hash_stable(hcx, hasher);
    for declaration in &resources.device_externs {
        declaration.stable_crate_id.hash_stable(hcx, hasher);
        declaration.export_name.hash_stable(hcx, hasher);
        declaration.signature.hash_stable(hcx, hasher);
        declaration.attrs.is_convergent.hash_stable(hcx, hasher);
        declaration.attrs.is_pure.hash_stable(hcx, hasher);
        declaration.attrs.is_readonly.hash_stable(hcx, hasher);
    }

    "families".hash_stable(hcx, hasher);
    resources.families.len().hash_stable(hcx, hasher);
    for family in &resources.families {
        family.root_symbol.hash_stable(hcx, hasher);
        family.root_export_name.hash_stable(hcx, hasher);
        family.function_symbols.hash_stable(hcx, hasher);
    }

    "core-index-trait".hash_stable(hcx, hasher);
    resources.core_index_trait.hash_stable(hcx, hasher);
}

fn inline_attr_tag(attribute: mir_importer::InlineAttr) -> &'static str {
    match attribute {
        mir_importer::InlineAttr::None => "none",
        mir_importer::InlineAttr::Hint => "hint",
        mir_importer::InlineAttr::Always => "always",
        mir_importer::InlineAttr::DeviceAlways => "device-always",
    }
}

fn hash_debug_source_scope_map(
    map: Option<&llvm_export::ops::DebugSourceScopeMap>,
    hcx: &mut StableHashingContext<'_>,
    hasher: &mut StableHasher,
) {
    map.is_some().hash_stable(hcx, hasher);
    let Some(map) = map else {
        return;
    };

    map.scopes.len().hash_stable(hcx, hasher);
    for scope in &map.scopes {
        scope.id.hash_stable(hcx, hasher);
        scope.parent.hash_stable(hcx, hasher);
        hash_debug_source_position(scope.span.as_ref(), hcx, hasher);
        scope.inlined.is_some().hash_stable(hcx, hasher);
        if let Some(inlined) = &scope.inlined {
            inlined.callee_name.hash_stable(hcx, hasher);
            hash_debug_source_position(inlined.callsite.as_ref(), hcx, hasher);
        }
    }

    map.locations.len().hash_stable(hcx, hasher);
    for location in &map.locations {
        hash_debug_source_position(Some(&location.pos), hcx, hasher);
        location.scope.hash_stable(hcx, hasher);
    }
}

fn hash_debug_source_position(
    position: Option<&llvm_export::ops::DebugSourcePosition>,
    hcx: &mut StableHashingContext<'_>,
    hasher: &mut StableHasher,
) {
    position.is_some().hash_stable(hcx, hasher);
    if let Some(position) = position {
        position
            .file
            .as_os_str()
            .as_encoded_bytes()
            .hash_stable(hcx, hasher);
        position.line.hash_stable(hcx, hasher);
        position.column.hash_stable(hcx, hasher);
    }
}

fn hash_successors(
    successors: &[Vec<usize>],
    hcx: &mut StableHashingContext<'_>,
    hasher: &mut StableHasher,
) {
    successors.len().hash_stable(hcx, hasher);
    for block in successors {
        block.len().hash_stable(hcx, hasher);
        for successor in block {
            (*successor as u64).hash_stable(hcx, hasher);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_fingerprint_hex_is_fixed_width_and_little_endian() {
        let fingerprint = DeviceSemanticFingerprint([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0xff,
        ]);
        assert_eq!(fingerprint.to_hex(), "000102030405060708090a0b0c0d0eff");
    }

    #[test]
    fn fingerprint_error_preserves_fail_closed_context() {
        let error = DeviceSemanticFingerprintError::new("unsupported allocation");
        assert_eq!(error.to_string(), "unsupported allocation");
    }
}
