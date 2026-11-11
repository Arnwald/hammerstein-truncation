//! Truncation error of single-level Hammerstein identification from
//! exponential swept sines: numerical companion of the paper.
//!
//! Three experiments, each writing `.npy` arrays and a `meta.json`:
//!
//! - [`exact`]: the exact prediction error for `rho arctan(x/rho)` and
//!   `rho artanh(x/rho)`, the general bounds, the low-level floors, the map
//!   over levels and radii, and the running of the effective kernels;
//! - [`chain`]: the same error for a filtered three-branch system, in closed
//!   form and through a simulated swept-sine measurement chain;
//! - [`ess`]: the deconvolved response of a ninth-order cascade to a sweep.
//!
//! Supporting modules: [`mixing`] (harmonic mixing matrix, single-level
//! estimator), [`statics`] (static nonlinearities, alias-free synthesis),
//! [`sweep`], [`spectral`] (deconvolution, harmonic separation), [`filters`],
//! [`linalg`], [`npy`], [`out`].

// Index loops mirror the formulas (sums over orders n and harmonics m).
#![allow(clippy::needless_range_loop)]

pub mod chain;
pub mod ess;
pub mod exact;
pub mod filters;
pub mod linalg;
pub mod mixing;
pub mod npy;
pub mod out;
pub mod spectral;
pub mod statics;
pub mod sweep;
