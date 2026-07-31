/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![cfg_attr(target_os = "cuda", no_std)]

use core::array;

// Aggregate helpers live in a separate crate to exercise cross-crate inline intent.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Vector(pub [f32; 3]);

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Matrix(pub [Vector; 3]);

#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct Vol<N: Copy>(pub [N; 2]);

#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct Face<N: Copy>(pub [N; 2]);

#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct Soln<N: Copy>(pub [N; 2]);

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PressureMatrix(pub [[f64; 3]; 3]);

#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct PressureVol(pub [PressureMatrix; 2]);

#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct PressureFace(pub [PressureMatrix; 2]);

#[inline(always)]
pub fn make_volume(base: f32, lane: f32) -> Vol<Vector> {
    Vol(array::from_fn(|node| {
        let node = node as f32;
        Vector([
            base + lane * 0.03125 + node,
            base * 0.5 - lane * 0.015625 + node * 2.0,
            base * 1.5 + lane * 0.0078125 - node,
        ])
    }))
}

#[inline(always)]
fn gradient(input: Vol<Vector>, scale: f32) -> Vol<Matrix> {
    Vol(array::from_fn(|node| {
        Matrix(array::from_fn(|row| {
            Vector(array::from_fn(|column| {
                let x = input.0[(node + row) & 1].0[column];
                let y = input.0[node].0[(column + row + 1) % 3];
                x * scale + y * (0.125 * (row + column + 1) as f32)
            }))
        }))
    }))
}

#[inline(always)]
fn volume_flux(input: Vol<Vector>, gradient: Vol<Matrix>, viscosity: f32) -> Vol<Vector> {
    Vol(array::from_fn(|node| {
        Vector(array::from_fn(|component| {
            let advective = input.0[node].0[component]
                * (input.0[node].0[0] + input.0[node].0[1] + input.0[node].0[2]);
            let diffusive = gradient.0[node].0[0].0[component]
                + gradient.0[node].0[1].0[component]
                + gradient.0[node].0[2].0[component];
            advective * 0.01 - viscosity * diffusive
        }))
    }))
}

#[inline(always)]
fn to_face(input: Vol<Vector>, normal_scale: f32) -> Face<Vector> {
    Face(array::from_fn(|node| {
        Vector(array::from_fn(|component| {
            let own = input.0[node].0[component];
            let neighbour = input.0[1 - node].0[(component + 1) % 3];
            own * normal_scale + neighbour * (1.0 - normal_scale)
        }))
    }))
}

#[inline(always)]
fn lift_face(face: Face<Vector>, penalty: f32) -> Vol<Vector> {
    Vol(array::from_fn(|node| {
        Vector(array::from_fn(|component| {
            let jump = face.0[node].0[component] - face.0[1 - node].0[component];
            face.0[node].0[component] + penalty * jump
        }))
    }))
}

#[inline(always)]
fn blend(left: Vol<Vector>, right: Vol<Vector>, scale: f32) -> Vol<Vector> {
    Vol(array::from_fn(|node| {
        Vector(array::from_fn(|component| {
            left.0[node].0[component] + scale * right.0[node].0[component]
        }))
    }))
}

#[inline(always)]
fn bulky_mix(mut state: Vol<Vector>, coefficient: f32) -> Vol<Vector> {
    macro_rules! mix {
        ($($tag:literal),+ $(,)?) => {
            $(
                {
                    let node = $tag & 1;
                    let component = ($tag / 2) % 3;
                    let other_node = 1 - node;
                    let other_component = (component + 1 + ($tag % 2)) % 3;
                    let own = state.0[node].0[component];
                    let other = state.0[other_node].0[other_component];
                    state.0[node].0[component] = own
                        * (0.999 + ($tag as f32) * 0.0000001)
                        + other * (0.00001 + ($tag as f32) * 0.00000001)
                        + coefficient * (0.000001 + ($tag as f32) * 0.000000001);
                }
            )+
        };
    }

    mix!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    );
    mix!(
        128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
        144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
        192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
        208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    );
    state
}

#[inline(always)]
fn combine(volume: Vol<Vector>, lifted: Vol<Vector>, mass: f32) -> Soln<Vector> {
    Soln(array::from_fn(|node| {
        Vector(array::from_fn(|component| {
            volume.0[node].0[component] * mass + lifted.0[node].0[component]
        }))
    }))
}

#[inline(always)]
pub fn apply_operator(input: Vol<Vector>, coefficient: f32) -> Soln<Vector> {
    let input = bulky_mix(input, coefficient);
    let gradient_x = gradient(input, coefficient * 0.25);
    let volume_x = volume_flux(input, gradient_x, coefficient * 0.03125);
    let face_x = to_face(input, coefficient * 0.0625 + 0.25);
    let lifted_x = lift_face(face_x, coefficient * 0.015625);
    let state_x = blend(volume_x, lifted_x, 0.5);

    let gradient_y = gradient(state_x, coefficient * 0.1875);
    let volume_y = volume_flux(state_x, gradient_y, coefficient * 0.0234375);
    let face_y = to_face(state_x, coefficient * 0.046875 + 0.375);
    let lifted_y = lift_face(face_y, coefficient * 0.01171875);
    let state_y = blend(volume_y, lifted_y, 0.375);

    let gradient_z = gradient(state_y, coefficient * 0.140625);
    let volume_z = volume_flux(state_y, gradient_z, coefficient * 0.017578125);
    let face_z = to_face(state_y, coefficient * 0.03515625 + 0.5);
    let lifted_z = lift_face(face_z, coefficient * 0.0087890625);
    let state_z = blend(volume_z, lifted_z, 0.25);

    combine(state_z, lifted_x, coefficient * 0.125 + 1.0)
}

#[inline(always)]
pub fn checksum(value: Soln<Vector>) -> f32 {
    let mut sum = 0.0;
    for node in 0..2 {
        for component in 0..3 {
            sum += value.0[node].0[component] * (1 + node * 3 + component) as f32;
        }
    }
    sum
}

#[inline(always)]
pub fn make_pressure_volume(base: f64, lane: f64) -> PressureVol {
    PressureVol(array::from_fn(|node| {
        PressureMatrix(array::from_fn(|row| {
            array::from_fn(|column| {
                let ordinal = (node * 9 + row * 3 + column) as f64;
                base * (1.0 + ordinal * 0.0078125)
                    + lane * (0.5 - ordinal * 0.00390625)
                    + ordinal * 0.03125
            })
        }))
    }))
}

#[inline(always)]
fn pressure_mix(mut state: PressureVol, coefficient: f64) -> PressureVol {
    macro_rules! mix {
        ($($tag:literal),+ $(,)?) => {
            $(
                {
                    let node = $tag & 1;
                    let row = ($tag / 2) % 3;
                    let column = ($tag / 6) % 3;
                    let other_node = 1 - node;
                    let other_row = (row + 1 + ($tag % 2)) % 3;
                    let other_column = (column + 2) % 3;
                    let own = state.0[node].0[row][column];
                    let other = state.0[other_node].0[other_row][other_column];
                    state.0[node].0[row][column] = own
                        * (0.999 + ($tag as f64) * 0.0000001)
                        + other * (0.00001 + ($tag as f64) * 0.00000001)
                        + coefficient * (0.000001 + ($tag as f64) * 0.000000001);
                }
            )+
        };
    }

    mix!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    );
    state
}

#[inline(always)]
fn pressure_face_coordinate(node: usize, axis: usize) -> usize {
    let lane = node * 17 + axis * 11;
    (lane ^ (lane >> 1)) & 3
}

#[inline(always)]
fn extract_pressure_face(axis: usize, volume: PressureVol) -> PressureFace {
    PressureFace(array::from_fn(|node| {
        let coordinate = pressure_face_coordinate(node, axis);
        let should_extract = coordinate == 0 || coordinate == 3;
        let mut value = volume.0[node];
        if !should_extract {
            for component in 0..9 {
                value.0[component / 3][component % 3] = 0.0;
            }
        }
        value
    }))
}

#[inline(always)]
fn pressure_face_checksum(face: PressureFace) -> f64 {
    let mut sum = 0.0;
    for node in 0..2 {
        for row in 0..3 {
            for column in 0..3 {
                sum += face.0[node].0[row][column]
                    * (1 + node * 9 + row * 3 + column) as f64;
            }
        }
    }
    sum
}

#[inline(always)]
pub fn pressure_operator(input: PressureVol, coefficient: f64) -> f64 {
    let first = pressure_mix(input, coefficient);
    let second = pressure_mix(first, coefficient + 0.125);
    let third = pressure_mix(second, coefficient + 0.25);
    let third_face = extract_pressure_face(2, third);
    pressure_face_checksum(third_face)
}
