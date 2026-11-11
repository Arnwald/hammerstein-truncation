//! Single-level prediction error of a filtered Hammerstein system, in closed
//! form and through a simulated swept-sine measurement chain.
//!
//! The test system (`sim/systems/static_arctan2.0.toml`) has a linear branch,
//! an odd branch `rho arctan(x/rho) - x` and an even branch
//! `(x/rho)^2 / (1 + (x/rho)^2)`, each followed by its own filter, with
//! rho = 2: a pure Hammerstein cascade with infinitely many orders, whose
//! kernels depend on frequency and whose harmonic responses are known in
//! closed form at every level.
//!
//! # Closed form ([`closed_form`])
//!
//! No signal is involved. At each frequency of the band, `H_m(f, lambda)` is
//! the sum of the closed-form harmonics of the three branches, each multiplied
//! by the response of its filter. The kernels follow from the harmonics at
//! `lambda = 1` by the single-level estimator, the prediction at lambda scales
//! the n-th kernel as `lambda^n`, and the error of each harmonic is reduced to
//! one number per gain, the energy of the error relative to that of the
//! harmonic over the band,
//! `sum_f |e_m(f, lambda)|^2 / sum_f |H_m(f, lambda)|^2`.
//!
//! # Measurement chain ([`run`])
//!
//! 1. Farina's sweep ([`crate::sweep`]), sampled at 96 kHz, up to 24 kHz.
//! 2. The system output is synthesized without aliasing
//!    ([`crate::statics::alias_free_static`]): the nonlinearities are never
//!    evaluated on the samples.
//! 3. The output is deconvolved with Farina's inverse filter, each harmonic
//!    response is cut out around `-L ln m`, aligned, rotated by the constant
//!    phase of Farina's sweep and windowed ([`crate::spectral::Slicer`]).
//! 4. The leak of the linear response into the windows of the harmonics, due
//!    to the pre-ringing of the inverse filter, is measured once by
//!    deconvolving the sweep itself (a loopback, no system) and subtracted
//!    from the harmonics measured at the identification level:
//!    `H_m <- H_m - H_1 L_m / L_1`.
//! 5. The kernels follow from the single-level estimator with N = 7.
//! 6. The prediction at each gain is the output of the identified cascade,
//!    `sum_n lambda^n g_n * x^n`, taken through the same chain: the kernels,
//!    windowed in time, are applied to the deconvolved alias-free powers of
//!    the sweep, and the result is cut out like a measurement.
//! 7. The truth at each gain is the output of the system driven at that gain,
//!    synthesized as in step 2 and processed as in step 3.
//! 8. The error is the same energy ratio as for the closed form, over the
//!    frequencies of each harmonic between 100 Hz and 10 kHz.
//!
//! # Outputs (directory `--out`, `output/chain` by default)
//!
//! - `closed_form.npy` `[gains, 7]`: closed-form error (dB);
//! - `closed_form_level.npy` `[gains, 7]`: energy of each true harmonic re
//!   `H_1` (dB), from the closed forms;
//! - `chain.npy` `[gains, 7]`: error through the measurement chain (dB);
//! - `meta.json`: gains, band, sweep parameters, `farina_phase_rad` (the
//!   constant phase `2 pi f_start L` of Farina's sweep, modulo 2 pi), and
//!   `leak_db`, the energy of the loopback in the windows 2 ... 7 re its
//!   linear window (dB).

use num_complex::Complex64 as C64;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::path::PathBuf;
use std::time::Instant;

use crate::exact::{from_db, grid};
use crate::mixing::{Spectra, mixing_matrix, single_level};
use crate::out::{Json, Out};
use crate::spectral::{
    Deconvolver, Fft, SliceWindow, Slicer, alias_free_power, analytic, hann_fade_at_peak,
    next_fast_len, segment, smooth_octave,
};
use crate::statics::StaticSystem;
use crate::sweep::{Sweep, SweepSpec};

const SYSTEM: &str = "static_arctan2.0";
const FS: f64 = 96_000.0;
/// End of the fade-in of the sweep (Hz); the sweep starts two octaves lower.
const SWEEP_F1: f64 = 1.0;
/// Requested end of the pass band: 0.6 of the Nyquist frequency, capped so
/// that the fade-out (to 1.8 times the end of the band) stays below 0.9 of
/// the Nyquist frequency: f2 = 24 kHz, sweep to 43.2 kHz.
const SWEEP_F2: f64 = FS / 2.0 * 0.6;
const SWEEP_F2_MARGIN: f64 = 1.8;
const SWEEP_CEILING: f64 = 0.9;
const DEFAULT_DURATION: f64 = 100.0;
/// Anti-aliasing filter of the converter: flat up to AA_PASS * FS, zero at FS / 2.
const AA_PASS: f64 = 0.44;
/// Harmonics kept, hence kernels identified.
const N_HARMONICS: usize = 7;
/// Length of the slices cut out of the deconvolved response.
const N_SLICE: usize = 1 << 17;
/// The window of a harmonic is capped by this fraction of the distance to its neighbours.
const SLICE_MARGIN: f64 = 0.7;
const SLICE_WINDOW: SliceWindow = SliceWindow {
    fade_in_ms: 100.0,
    pre_peak_ms: 400.0,
    post_peak_ms: 400.0,
    fade_out_ms: 100.0,
};
/// Time window of the identified kernels before they are used to predict.
const KERNEL_WINDOW: SliceWindow = SliceWindow {
    fade_in_ms: 100.0,
    pre_peak_ms: 400.0,
    post_peak_ms: 400.0,
    fade_out_ms: 100.0,
};
/// Band over which the errors are evaluated (Hz).
const ERROR_BAND: (f64, f64) = (100.0, 10_000.0);
/// The band stops at least this fraction of an octave below the end of the sweep band.
const EDGE_OCT: f64 = 0.25;
/// Prediction gains (dB re the identification level).
const GAINS_DB: (f64, f64, f64) = (-30.0, 12.0, 1.0);
/// Smoothing of the loopback energies used for the leak figure (octave).
const LEAK_SMOOTH_OCT: f64 = 1.0 / 3.0;
/// Floor of the reported errors (dB).
const ERROR_FLOOR_DB: f64 = -200.0;

fn db10(x: f64) -> f64 {
    10.0 * x.max(1e-300).log10()
}

/// Energy of `est - reference` relative to that of `reference` over the bins
/// of `band` (dB).
fn nmse_db(est: &[C64], reference: &[C64], band: &[usize]) -> f64 {
    let num: f64 = band
        .iter()
        .map(|&f| (est[f] - reference[f]).norm_sqr())
        .sum();
    let den: f64 = band.iter().map(|&f| reference[f].norm_sqr()).sum();
    (db10(num) - db10(den)).max(ERROR_FLOOR_DB)
}

/// Single-level prediction error of a filtered system, in closed form, per
/// gain and per harmonic: `10 log10(sum |H_hat - H|^2 / sum |H|^2)` over the
/// band; and the energy of each harmonic re `H_1`,
/// `10 log10(sum |H_m|^2 / sum |H_1|^2)`. Both `[gain][harmonic]`.
pub fn closed_form(sys: &StaticSystem, gains_db: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = N_HARMONICS;
    let df = FS / N_SLICE as f64;
    let freqs: Vec<f64> = (0..=N_SLICE / 2)
        .map(|k| k as f64 * df)
        .filter(|&f| f >= ERROR_BAND.0 && f <= ERROR_BAND.1)
        .collect();
    let a = mixing_matrix(m);
    let g = single_level(&sys.harmonics(m, &freqs, 1.0), &a);
    gains_db
        .par_iter()
        .map(|&gdb| {
            let lam = from_db(gdb);
            let truth = sys.harmonics(m, &freqs, lam);
            let energy: Vec<f64> = truth
                .iter()
                .map(|t| t.iter().map(|v| v.norm_sqr()).sum())
                .collect();
            let err = (0..m)
                .map(|hm| {
                    let num: f64 = truth[hm]
                        .iter()
                        .enumerate()
                        .map(|(k, t)| {
                            let pred: C64 = (hm..m)
                                .map(|n| a[n][hm] * lam.powi(n as i32 + 1) * g[n][k])
                                .sum();
                            (pred - t).norm_sqr()
                        })
                        .sum();
                    10.0 * (num / energy[hm]).log10()
                })
                .collect();
            let level = energy
                .iter()
                .map(|e| 10.0 * (e / energy[0]).log10())
                .collect();
            (err, level)
        })
        .unzip()
}

/// Spectra of the segments of a deconvolved power around each harmonic
/// window, for the local convolutions of [`predict_orders`].
fn window_segments(d: &[f64], slicer: &Slicer, conv_fft: &Fft) -> Vec<Vec<C64>> {
    let n_slice = slicer.n as isize;
    (0..slicer.m)
        .map(|k| {
            // Y[s + q] = sum_i h[i] D[s + q + shft - i]: a segment of D of length 2N - 1
            let start = slicer.start(k) + slicer.shft as isize - (n_slice - 1);
            conv_fft.forward(&segment(d, start, 2 * slicer.n - 1))
        })
        .collect()
}

/// Harmonic responses predicted by each order of the identified cascade at
/// the identification level: `[order][harmonic][bin < n_bins]`. The kernel of
/// order n (a time response with its peak at `shft`) is convolved with the
/// deconvolved power `x^n` around each window, and the result is cut out like
/// a measurement.
fn predict_orders(
    kernels_td: &[Vec<f64>],
    segments: &[Vec<Vec<C64>>],
    slicer: &Slicer,
    conv_fft: &Fft,
    n_bins: usize,
) -> Vec<Spectra> {
    let n = slicer.n;
    kernels_td
        .par_iter()
        .enumerate()
        .map(|(i, h)| {
            let h_spec = conv_fft.forward(h);
            (0..slicer.m)
                .map(|k| {
                    let prod: Vec<C64> = h_spec
                        .iter()
                        .zip(&segments[i][k])
                        .map(|(a, b)| a * b)
                        .collect();
                    let y = conv_fft.inverse(prod);
                    let mut spec = slicer.finish(k, y[n - 1..2 * n - 1].to_vec());
                    spec.truncate(n_bins);
                    spec
                })
                .collect()
        })
        .collect()
}

/// Harmonics at a gain: `sum_n lambda^n B[n - 1]` (`B`: `[order][harmonic][bin]`).
fn at_gain(b: &[Spectra], lam: f64) -> Spectra {
    let (m_len, f_len) = (b[0].len(), b[0][0].len());
    let mut out = vec![vec![C64::new(0.0, 0.0); f_len]; m_len];
    for (i, bn) in b.iter().enumerate() {
        let w = lam.powi(i as i32 + 1);
        for m in 0..m_len {
            for f in 0..f_len {
                out[m][f] += bn[m][f] * w;
            }
        }
    }
    out
}

pub fn run(argv: Vec<String>) {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../output/chain");
    let mut duration = DEFAULT_DURATION;
    let mut integer_k = true;
    let mut it = argv.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => dir = PathBuf::from(it.next().expect("--out DIR")),
            "--duration" => {
                duration = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--duration SECONDS")
            }
            "--free-f1" => integer_k = false,
            _ => panic!("unknown option: {a}"),
        }
    }
    let out = Out::new(dir);
    let t_start = Instant::now();
    let stamp = || format!("[{:6.1} s]", t_start.elapsed().as_secs_f64());
    let m = N_HARMONICS;
    let gains = grid(GAINS_DB);
    let sys = StaticSystem::standard(SYSTEM);

    // Closed form
    let (closed, level) = closed_form(&sys, &gains);
    out.f64_rows("closed_form.npy", &closed);
    out.f64_rows("closed_form_level.npy", &level);
    println!("{} closed form at {} gains", stamp(), gains.len());

    // Sweep, deconvolution, slicing
    let sweep = Sweep::new(SweepSpec {
        fs: FS,
        f1: SWEEP_F1,
        duration,
        f2: SWEEP_F2,
        margin: SWEEP_F2_MARGIN,
        ceiling: SWEEP_CEILING,
        integer_k,
    });
    let k = sweep.f1_sweep * sweep.l_param;
    println!(
        "{} sweep: {:.4} Hz to {:.0} Hz (pass band {:.2} Hz to {:.0} Hz), L = {:.4} s, \
         f_start L = {:.6}, N = {}",
        stamp(),
        sweep.f1_sweep,
        sweep.f2_cutoff,
        sweep.f1,
        sweep.f2,
        sweep.l_param,
        k,
        sweep.n_samples
    );
    let deconv = Deconvolver::new(&sweep.inverse, sweep.n_samples);
    let slicer = Slicer::new(&sweep, m, N_SLICE, SLICE_MARGIN, &SLICE_WINDOW);
    let shft = slicer.shft;
    let conv_fft = Fft::new(next_fast_len(3 * N_SLICE - 2));
    let slice_fft = Fft::new(N_SLICE);
    let f_len = N_SLICE / 2 + 1;
    let freqs: Vec<f64> = (0..f_len).map(|i| i as f64 * FS / N_SLICE as f64).collect();
    let n_bins = freqs.iter().filter(|&&f| f <= sweep.f2).count();
    let band_hi = ERROR_BAND.1.min(sweep.f2 * 2f64.powf(-EDGE_OCT));
    let band: Vec<usize> = (0..n_bins)
        .filter(|&i| freqs[i] >= ERROR_BAND.0 && freqs[i] <= band_hi)
        .collect();
    let z = analytic(&sweep.windowed);

    // Deconvolved alias-free powers x^1 ... x^N around each window, and the
    // loopback (the sweep itself through the chain, no system)
    let mut segments = Vec::with_capacity(m);
    let mut loopback = Vec::new();
    for n in 1..=m {
        let d = deconv.deconvolve(&alias_free_power(&z, n, &sweep.f_inst, FS, AA_PASS));
        segments.push(window_segments(&d, &slicer, &conv_fft));
        if n == 1 {
            loopback = slicer.spectra(&d);
        }
    }
    println!("{} powers x^1..x^{m} deconvolved", stamp());

    // Leak of the linear response into the windows of the harmonics
    let power = |s: &[C64]| -> Vec<f64> {
        smooth_octave(
            &s.iter().map(|v| v.norm_sqr()).collect::<Vec<_>>(),
            LEAK_SMOOTH_OCT,
        )
    };
    let p0 = power(&loopback[0]);
    let leak_db: Vec<f64> = (1..m)
        .map(|h| {
            let p = power(&loopback[h]);
            db10(band.iter().map(|&f| p[f] / p0[f].max(1e-300)).sum::<f64>() / band.len() as f64)
        })
        .collect();
    println!(
        "{} loopback leak in the windows 2..{m} (dB re H1): {}",
        stamp(),
        leak_db
            .iter()
            .map(|v| format!("{v:.1}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    // The system driven at a gain, through the chain: [harmonic][bin]
    let measure = |lam: f64| -> Spectra {
        let y = sys.respond(&z, &sweep.f_inst, FS, AA_PASS, lam);
        slicer.spectra(&deconv.deconvolve(&y))
    };

    // Identification at the reference level, leak of the linear response removed
    let mut h_ref = measure(1.0);
    for hm in 1..m {
        for f in 0..f_len {
            if loopback[0][f].norm() > 1e-12 {
                let leak = h_ref[0][f] * loopback[hm][f] / loopback[0][f];
                h_ref[hm][f] -= leak;
            }
        }
    }
    let a = mixing_matrix(m);
    let kernels = single_level(&h_ref, &a);
    let ms = |v: f64| (v * FS / 1000.0).round() as usize;
    let kernels_td: Vec<Vec<f64>> = kernels
        .iter()
        .map(|g| {
            hann_fade_at_peak(
                &slice_fft.inverse(g.clone()),
                shft,
                ms(KERNEL_WINDOW.fade_in_ms),
                ms(KERNEL_WINDOW.pre_peak_ms),
                ms(KERNEL_WINDOW.post_peak_ms),
                ms(KERNEL_WINDOW.fade_out_ms),
            )
        })
        .collect();
    let predicted = predict_orders(&kernels_td, &segments, &slicer, &conv_fft, n_bins);
    println!("{} kernels identified at the reference level", stamp());

    // Prediction against the truth at each gain (one gain at a time: the
    // synthesis is parallel over the samples, and the memory stays bounded)
    let mut err = Vec::with_capacity(gains.len());
    for &g in &gains {
        let lam = from_db(g);
        let truth = measure(lam);
        let pred = at_gain(&predicted, lam);
        let row: Vec<f64> = (0..m)
            .map(|hm| nmse_db(&pred[hm], &truth[hm], &band))
            .collect();
        println!(
            "{} {g:+5.0} dB: {}",
            stamp(),
            row.iter()
                .map(|v| format!("{v:7.1}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        err.push(row);
    }
    out.f64_rows("chain.npy", &err);

    Json::new()
        .str("experiment", "chain")
        .str("system", SYSTEM)
        .num("fs", FS)
        .num("duration", duration)
        .num("f1_sweep", sweep.f1_sweep)
        .num("f1", sweep.f1)
        .num("f2", sweep.f2)
        .num("f2_cutoff", sweep.f2_cutoff)
        .num("L", sweep.l_param)
        .num("f_start_L", k)
        .bool("integer_k", integer_k)
        .num("farina_phase_rad", 2.0 * PI * (k - k.round()))
        .int("harmonics", m as i64)
        .int("n_slice", N_SLICE as i64)
        .nums("gains_db", &gains)
        .nums("band_hz", &[ERROR_BAND.0, band_hi])
        .nums("leak_db", &leak_db)
        .write(&out, "meta.json");
    println!("{} output: {}", stamp(), out.dir.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closed-form error vanishes at the identification level and tends
    /// to the relative bias of the kernel of the same order at low level.
    #[test]
    fn closed_form_limits() {
        let sys = StaticSystem::standard(SYSTEM);
        let (err, level) = closed_form(&sys, &[-60.0, -50.0, 0.0]);
        assert!(err[2].iter().all(|&e| e < -250.0), "{:?}", err[2]);
        // a floor: 10 dB lower in level, same relative error
        for hm in 0..N_HARMONICS {
            assert!((err[0][hm] - err[1][hm]).abs() < 0.01, "m = {}", hm + 1);
        }
        // the seventh harmonic of the odd branch (arctan, rho = 2): -9.6 dB
        assert!((err[0][6] + 9.6).abs() < 0.1, "{}", err[0][6]);
        assert_eq!(level[0][0], 0.0);
    }
}
