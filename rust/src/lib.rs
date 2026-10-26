//! Fast substring search with AVX-512: a port of
//! [lemire/simdsearch](https://github.com/lemire/simdsearch).
//!
//! [`find`] is Guardzilla. [`needlehammer::find`] adds a branch-free second
//! scan for inputs that defeat the branch predictor and needs AVX-512 VBMI2
//! and VL. Every search takes O(n + m) time.
//!
//! The instruction set is chosen at compile time: build with
//! `-C target-cpu=native` or `-C target-feature=+avx512f,+avx512bw`.

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
compile_error!(
    "simdsearch needs x86-64 with AVX-512F and AVX-512BW: build with RUSTFLAGS=\"-C target-cpu=native\""
);

mod anchors;
mod byte;
mod simd;
mod verify;

pub mod guardzilla;
#[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
pub mod needlehammer;
pub mod twoway;

pub use guardzilla::find;
