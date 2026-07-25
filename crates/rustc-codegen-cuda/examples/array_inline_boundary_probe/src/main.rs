use cuda_device::kernel;

#[kernel]
pub fn closure_result_boundary(output: *mut u32) {
    let produced = core::array::from_fn::<_, 2, _>(|index| {
        let captured = index as u32;
        move |value: u32| {
            let mut result = value.wrapping_add(captured);
            for factor in [3, 5, 7, 11, 13, 17, 19, 23] {
                result = result.wrapping_mul(factor).wrapping_add(1);
            }
            result
        }
    });
    let value = produced[0](7) ^ produced[1](11);
    // SAFETY: this compiler fixture is not launched.
    unsafe { output.write(value) };
}

mod array {
    #[inline(never)]
    pub fn from_fn<const N: usize, F: FnMut(usize) -> u32>(mut callback: F) -> [u32; N] {
        core::array::from_fn(|index| callback(index))
    }
}

#[kernel]
pub fn user_suffix_boundary(output: *mut u32) {
    let values = array::from_fn::<2, _>(|index| index as u32 + 1);
    // SAFETY: this compiler fixture is not launched.
    unsafe { output.write(values[0] + values[1]) };
}

#[kernel]
pub fn oversized_capture_boundary(output: *mut u32, seed: u32) {
    let captured = [seed; 80];
    let values =
        core::array::from_fn::<_, 2, _>(move |index| captured[index].wrapping_add(index as u32));
    // SAFETY: this compiler fixture is not launched.
    unsafe { output.write(values[0] ^ values[1]) };
}

fn item_callback(index: usize) -> u32 {
    index as u32 + 17
}

#[kernel]
pub fn function_item_boundary(output: *mut u32) {
    let values = core::array::from_fn::<_, 2, _>(item_callback);
    // SAFETY: this compiler fixture is not launched.
    unsafe { output.write(values[0] + values[1]) };
}

#[kernel]
pub fn shared_callback_boundary(output: *mut u32, seed: u32) {
    let callback = |index| seed.wrapping_add(index as u32);
    let accepted_use = core::array::from_fn::<_, 2, _>(callback);
    let rejected_use = core::array::from_fn::<_, 129, _>(callback);
    // SAFETY: this compiler fixture is not launched.
    unsafe { output.write(accepted_use[1] ^ rejected_use[128]) };
}

fn main() {
    println!("SUCCESS: array inline boundaries compiled");
}
