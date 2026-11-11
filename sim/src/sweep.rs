//! Farina's exponential swept sine and its inverse filter.
//!
//! The sweep is `x(t) = w(f(t)) sin(phi(t))`, with
//! `phi(t) = 2 pi f_start L (exp(t/L) - 1)` and instantaneous frequency
//! `f(t) = f_start exp(t/L)`, `L = T / ln(f_end / f_start)`. The window `w`
//! acts on the instantaneous frequency: a sin^2 fade-in over two octaves, from
//! `f_start = f1/4` to `f1`, and a sin^2 fade-out from `f2` to `f_end`.
//!
//! The m-th harmonic of the sweep is the sweep itself, ahead by `L ln m`, up
//! to the constant phase `2 pi f_start L (m - 1)`. When `f_start L` is an
//! integer that phase is a multiple of 2 pi and the sweep coincides with the
//! synchronized sweep of Novak et al.; [`SweepSpec::integer_k`] adjusts
//! `f_start` so that it is.

use std::f64::consts::PI;

/// Parameters of a sweep.
#[derive(Clone, Copy, Debug)]
pub struct SweepSpec {
    /// Sampling rate (Hz).
    pub fs: f64,
    /// End of the fade-in (Hz); the sweep starts two octaves lower.
    pub f1: f64,
    /// Duration (s).
    pub duration: f64,
    /// Requested end of the pass band (Hz).
    pub f2: f64,
    /// The fade-out ends at `f2 * margin`.
    pub margin: f64,
    /// Fraction of the Nyquist frequency that the end of the sweep must not exceed.
    pub ceiling: f64,
    /// Adjust the start frequency so that `f_start * L` is an integer.
    pub integer_k: bool,
}

pub struct Sweep {
    pub fs: f64,
    /// End of the fade-in.
    pub f1: f64,
    /// Start of the sweep (`f_start` above).
    pub f1_sweep: f64,
    /// End of the pass band.
    pub f2: f64,
    /// End of the sweep (end of the fade-out).
    pub f2_cutoff: f64,
    /// Rate parameter L (s).
    pub l_param: f64,
    pub n_samples: usize,
    /// The sweep as played (windowed).
    pub windowed: Vec<f64>,
    /// Inverse filter: the sweep without its high-pass window, time-reversed,
    /// with an exponentially decaying envelope, calibrated.
    pub inverse: Vec<f64>,
    /// Instantaneous frequency of the sweep, sample by sample.
    pub f_inst: Vec<f64>,
}

/// Start frequency close to `f1_initial` for which `k = T f1 / ln(f2/f1)` is an
/// integer (bisection on a monotonic function).
fn adjust_f1_for_integer_k(f1_initial: f64, f2_sweep: f64, t: f64) -> f64 {
    let compute_k = |f1: f64| {
        let ln_r = if f2_sweep / f1 > 0.0 {
            (f2_sweep / f1).ln()
        } else {
            1e-12
        };
        t * f1 / ln_r
    };
    let k_target = compute_k(f1_initial).round().max(1.0);
    let f1_min = (f1_initial * 0.0001).max(0.0001);
    let f1_max = (f1_initial * 1000.0).min(f2_sweep * 0.999);
    let (k_min, k_max) = (compute_k(f1_min), compute_k(f1_max));
    if !(k_min.min(k_max) <= k_target && k_target <= k_min.max(k_max)) {
        return f1_initial;
    }
    let f1_test = 0.5 * (f1_min + f1_max);
    let increasing = compute_k(f1_test * 1.001) > compute_k(f1_test);
    let (mut lo, mut hi) = if increasing {
        (f1_min, f1_max)
    } else {
        (f1_max, f1_min)
    };
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let km = compute_k(mid);
        if (km - k_target).abs() < 1e-9 {
            return mid;
        }
        let below = if increasing {
            km < k_target
        } else {
            km > k_target
        };
        if below {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Window of the sweep at the instantaneous frequency f: sin^2 fades (in
/// octaves) from `f1_sweep` to `f1` and from `f2` to `f2_sweep`; with
/// `high_pass`, a fourth-order Butterworth high-pass modulus at `f1` as well
/// (only if `f1 >= 10` Hz).
fn window(f: f64, f1_sweep: f64, f1: f64, f2: f64, f2_sweep: f64, high_pass: bool) -> f64 {
    if !(f >= f1_sweep && f < f2_sweep) {
        return 0.0;
    }
    let bw = if high_pass && f1 >= 10.0 {
        1.0 / (1.0 + (f1 / f).powi(8)).sqrt()
    } else {
        1.0
    };
    if f1 > f1_sweep && f < f1 {
        let t = (f / f1_sweep).log2() / (f1 / f1_sweep).log2();
        return bw * (PI / 2.0 * t).sin().powi(2);
    }
    if f <= f2 {
        return bw;
    }
    let t = (f / f2).log2() / (f2_sweep / f2).log2();
    bw * (1.0 - (PI / 2.0 * t).sin().powi(2))
}

impl Sweep {
    pub fn new(spec: SweepSpec) -> Self {
        let SweepSpec {
            fs,
            f1: f1_user,
            duration,
            f2: f2_target,
            margin,
            ceiling,
            integer_k,
        } = spec;
        let nyquist = fs / 2.0;
        let f2 = f2_target.min(nyquist * ceiling / margin);
        let f2_cutoff = (f2 * margin).min(nyquist * ceiling);
        let (f1_sweep, f1) = if integer_k {
            let f = adjust_f1_for_integer_k(f1_user.max(0.1) / 4.0, f2_cutoff, duration);
            (f, f * 4.0)
        } else {
            (f1_user.max(0.1) / 4.0, f1_user)
        };
        let l_param = duration / (f2_cutoff / f1_sweep).ln();
        let n = (duration * fs).round() as usize;

        let mut windowed = vec![0.0; n];
        let mut no_window = vec![0.0; n];
        let mut f_inst_all = vec![0.0; n];
        for i in 0..n {
            let t = i as f64 / fs;
            let f_inst = f1_sweep * (t / l_param).exp();
            f_inst_all[i] = f_inst;
            let s = (2.0 * PI * f1_sweep * l_param * ((t / l_param).exp() - 1.0)).sin();
            windowed[i] = s * window(f_inst, f1_sweep, f1, f2, f2_cutoff, true);
            no_window[i] = s * window(f_inst, f1_sweep, f1, f2, f2_cutoff, false);
        }

        let mut inverse: Vec<f64> = (0..n)
            .map(|i| no_window[n - 1 - i] * (-(i as f64 / fs) / l_param).exp())
            .collect();
        // Calibration: |X(f) Inv(f)| = 1 at one frequency of the pass band
        // (direct evaluation of the two discrete-time Fourier transforms)
        let measure_freq = (f1 * f2).sqrt().max(300f64.min(f2)).min(f2);
        let w = 2.0 * PI * measure_freq / fs;
        let dtft = |x: &[f64]| {
            let (mut re, mut im) = (0.0, 0.0);
            for (t, v) in x.iter().enumerate() {
                let (s, c) = (w * t as f64).sin_cos();
                re += v * c;
                im -= v * s;
            }
            (re * re + im * im).sqrt()
        };
        let mag = dtft(&windowed) * dtft(&inverse);
        if mag > 1e-10 {
            for v in inverse.iter_mut() {
                *v /= mag;
            }
        }
        Sweep {
            fs,
            f1,
            f1_sweep,
            f2,
            f2_cutoff,
            l_param,
            n_samples: n,
            windowed,
            inverse,
            f_inst: f_inst_all,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_k_makes_the_sweep_synchronized() {
        let spec = SweepSpec {
            fs: 96_000.0,
            f1: 1.0,
            duration: 5.0,
            f2: 24_000.0,
            margin: 1.8,
            ceiling: 0.9,
            integer_k: true,
        };
        let s = Sweep::new(spec);
        let k = s.f1_sweep * s.l_param;
        assert!((k - k.round()).abs() < 1e-8, "k = {k}");
        assert_eq!(s.f2, 24_000.0);
        let free = Sweep::new(SweepSpec {
            integer_k: false,
            ..spec
        });
        assert_eq!(free.f1_sweep, 0.25);
        let k = free.f1_sweep * free.l_param;
        assert!((k - k.round()).abs() > 1e-3, "k = {k}");
    }
}
