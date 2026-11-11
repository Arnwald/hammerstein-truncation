//! Real FFTs, deconvolution of the sweep, separation of the harmonic
//! responses, analytic signal, and alias-free powers of a sweep.

use num_complex::Complex64 as C64;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f64::consts::PI;
use std::sync::Arc;

use crate::sweep::Sweep;

/// Smallest 5-smooth integer >= n (fast FFT sizes).
pub fn next_fast_len(n: usize) -> usize {
    let mut m = n.max(1);
    loop {
        let mut k = m;
        for p in [2, 3, 5] {
            while k.is_multiple_of(p) {
                k /= p;
            }
        }
        if k == 1 {
            return m;
        }
        m += 1;
    }
}

/// A pair of real FFT plans of fixed size (shareable between threads).
pub struct Fft {
    pub n: usize,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
}

impl Fft {
    pub fn new(n: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        Fft {
            n,
            r2c: planner.plan_fft_forward(n),
            c2r: planner.plan_fft_inverse(n),
        }
    }

    /// Spectrum (n/2 + 1 bins) of `x`, zero-padded to n.
    pub fn forward(&self, x: &[f64]) -> Vec<C64> {
        let mut buf = vec![0.0; self.n];
        let len = x.len().min(self.n);
        buf[..len].copy_from_slice(&x[..len]);
        let mut out = self.r2c.make_output_vec();
        self.r2c.process(&mut buf, &mut out).expect("rfft");
        out
    }

    /// Normalized inverse transform; the imaginary parts of the DC and Nyquist
    /// bins are ignored, as in `numpy.fft.irfft`.
    pub fn inverse(&self, mut spec: Vec<C64>) -> Vec<f64> {
        spec[0].im = 0.0;
        if self.n.is_multiple_of(2) {
            let last = spec.len() - 1;
            spec[last].im = 0.0;
        }
        let mut out = self.c2r.make_output_vec();
        self.c2r.process(&mut spec, &mut out).expect("irfft");
        let scale = 1.0 / self.n as f64;
        out.iter_mut().for_each(|v| *v *= scale);
        out
    }
}

/// Deconvolution of the sweep: linear convolution with Farina's inverse
/// filter (the time-reversed sweep with an exponential envelope).
///
/// The linear response is compressed into an impulse response at index
/// `inverse.len() - 1` of the output, and the response to the m-th harmonic
/// into an impulse response `L ln m` earlier. The product of the sweep and of
/// its inverse filter is 1 only in the stationary-phase approximation: its
/// impulse response has low-level ripples on both sides, so the linear
/// response slightly "pre-rings" at negative times, where the harmonics are.
/// This is the deconvolution leak that the chain measures with a loopback
/// (the sweep deconvolved with no system) and subtracts.
pub struct Deconvolver {
    fft: Fft,
    inv_spec: Vec<C64>,
    inv_len: usize,
}

impl Deconvolver {
    /// `signal_len`: length of the signals to deconvolve.
    pub fn new(inverse: &[f64], signal_len: usize) -> Self {
        let fft = Fft::new(next_fast_len(signal_len + inverse.len() - 1));
        let inv_spec = fft.forward(inverse);
        Deconvolver {
            fft,
            inv_spec,
            inv_len: inverse.len(),
        }
    }

    /// `y * inverse` (full linear convolution, `y.len() + inverse.len() - 1` samples).
    pub fn deconvolve(&self, y: &[f64]) -> Vec<f64> {
        assert!(
            y.len() + self.inv_len - 1 <= self.fft.n,
            "signal too long for the plan"
        );
        let mut spec = self.fft.forward(y);
        spec.iter_mut()
            .zip(&self.inv_spec)
            .for_each(|(s, i)| *s *= i);
        let mut out = self.fft.inverse(spec);
        out.truncate(y.len() + self.inv_len - 1);
        out
    }
}

/// `len` samples of `x` from `start` on (zeros outside the signal).
pub fn segment(x: &[f64], start: isize, len: usize) -> Vec<f64> {
    let mut out = vec![0.0; len];
    for (j, o) in out.iter_mut().enumerate() {
        let i = start + j as isize;
        if i >= 0 && (i as usize) < x.len() {
            *o = x[i as usize];
        }
    }
    out
}

/// Time window around a peak: fade-in, flat part before and after the peak,
/// fade-out (durations in ms).
pub struct SliceWindow {
    pub fade_in_ms: f64,
    pub pre_peak_ms: f64,
    pub post_peak_ms: f64,
    pub fade_out_ms: f64,
}

/// sin^2 window around a peak of known position: `fi` samples of fade-in,
/// `pre` flat samples before the peak, `post` after, `fo` of fade-out; zero
/// elsewhere.
pub fn hann_fade_at_peak(
    sig: &[f64],
    peak: usize,
    fi: usize,
    pre: usize,
    post: usize,
    fo: usize,
) -> Vec<f64> {
    let n = sig.len();
    let mut out = vec![0.0; n];
    let fi_start = peak.saturating_sub(pre + fi);
    let fi_end = fi_start.max(peak.saturating_sub(pre));
    if fi_end > fi_start {
        let len = (fi_end - fi_start) as f64;
        for (q, i) in (fi_start..fi_end).enumerate() {
            out[i] = sig[i] * 0.5 * (1.0 - (PI * q as f64 / len).cos());
        }
    }
    let fo_start = n.min(peak + post);
    out[fi_end..fo_start.max(fi_end)].copy_from_slice(&sig[fi_end..fo_start.max(fi_end)]);
    let fo_end = n.min(fo_start + fo);
    if fo_end > fo_start {
        let len = (fo_end - fo_start) as f64;
        for (q, i) in (fo_start..fo_end).enumerate() {
            out[i] = sig[i] * 0.5 * (1.0 + (PI * q as f64 / len).cos());
        }
    }
    out
}

/// Separation of the harmonic responses of a deconvolved signal.
///
/// The response to the k-th harmonic sits `L ln k` before the linear one. Each
/// is cut out as a slice of fixed length `n` with its peak at index `shft`,
/// aligned to the sub-sample (fractional delay applied in the frequency
/// domain), rotated by the constant phase `2 pi f_start L (k - 1)` of
/// Farina's sweep (trivial when `f_start L` is an integer), and windowed. The
/// window of each harmonic is capped by a fraction `margin` of the distance to
/// its neighbours.
pub struct Slicer {
    /// Slice length.
    pub n: usize,
    /// Index of the peak in a slice.
    pub shft: usize,
    /// Number of harmonics kept.
    pub m: usize,
    starts: Vec<isize>,
    frac: Vec<f64>,
    phi: Vec<f64>,
    win: Vec<[usize; 4]>,
    fft: Fft,
}

impl Slicer {
    /// `m` harmonics are kept; the (m+1)-th only serves to cap the window of the last one.
    pub fn new(sweep: &Sweep, m: usize, n: usize, margin: f64, w: &SliceWindow) -> Self {
        let fs = sweep.fs;
        let shft = n / 2;
        let t0 = sweep.n_samples as isize - 1;
        let max_h = m + 1;
        let dt: Vec<f64> = (1..=max_h)
            .map(|k| {
                if k == 1 {
                    0.0
                } else {
                    sweep.l_param * (k as f64).ln() * fs
                }
            })
            .collect();
        let mut starts = Vec::new();
        let mut frac = Vec::new();
        let mut phi = Vec::new();
        let mut win = Vec::new();
        let ms = |v: f64| (v * fs / 1000.0).round() as usize;
        for k_idx in 0..max_h {
            let dt_round = dt[k_idx].round();
            starts.push(t0 - dt_round as isize - shft as isize);
            frac.push(dt[k_idx] - dt_round);
            phi.push(2.0 * PI * sweep.f1_sweep * sweep.l_param * k_idx as f64);
            let mut before = if k_idx + 1 < max_h {
                dt[k_idx + 1] - dt[k_idx]
            } else {
                shft as f64
            };
            let mut after = if k_idx > 0 {
                dt[k_idx] - dt[k_idx - 1]
            } else {
                (n - shft) as f64
            };
            before = (before * margin).min(shft as f64);
            after = (after * margin).min((n - shft) as f64);
            let (before, after) = (before as usize, after as usize);
            let fi = ms(w.fade_in_ms).min(before);
            let pre = ms(w.pre_peak_ms).min(before.saturating_sub(fi));
            let post = ms(w.post_peak_ms).min(after);
            let fo = ms(w.fade_out_ms).min(after.saturating_sub(post));
            win.push([fi, pre, post, fo]);
        }
        Slicer {
            n,
            shft,
            m,
            starts,
            frac,
            phi,
            win,
            fft: Fft::new(n),
        }
    }

    /// Index, in the deconvolved response, of the first sample of slice k.
    pub fn start(&self, k: usize) -> isize {
        self.starts[k]
    }

    /// Raw slice -> fractional delay and phase rotation -> window -> spectrum.
    pub fn finish(&self, k: usize, raw: Vec<f64>) -> Vec<C64> {
        let (frac, phi) = (self.frac[k], self.phi[k]);
        let shifted = if frac.abs() >= 1e-9 || phi != 0.0 {
            let mut spec = self.fft.forward(&raw);
            let rot = C64::from_polar(1.0, phi);
            for (b, s) in spec.iter_mut().enumerate() {
                *s *= C64::from_polar(1.0, -2.0 * PI * frac * b as f64 / self.n as f64) * rot;
            }
            self.fft.inverse(spec)
        } else {
            raw
        };
        let [fi, pre, post, fo] = self.win[k];
        self.fft
            .forward(&hann_fade_at_peak(&shifted, self.shft, fi, pre, post, fo))
    }

    /// Spectra of the m harmonic responses of a deconvolved signal.
    pub fn spectra(&self, ir: &[f64]) -> Vec<Vec<C64>> {
        (0..self.m)
            .map(|k| self.finish(k, segment(ir, self.starts[k], self.n)))
            .collect()
    }
}

/// Mean over contiguous bands of `frac` octave (linear bins).
pub fn smooth_octave(power: &[f64], frac: f64) -> Vec<f64> {
    let n = power.len();
    let mut edges = vec![0usize, 1];
    while *edges.last().unwrap() < n {
        let e = *edges.last().unwrap();
        edges.push((e + 1).max((e as f64 * 2f64.powf(frac)).ceil() as usize));
    }
    edges.pop();
    let mut out = vec![0.0; n];
    for (i, &a) in edges.iter().enumerate() {
        let b = if i + 1 < edges.len() { edges[i + 1] } else { n };
        let mean = power[a..b].iter().sum::<f64>() / (b - a) as f64;
        out[a..b].iter_mut().for_each(|v| *v = mean);
    }
    out
}

/// Analytic signal `z = x + i H(x)`, Hilbert transform by real FFT
/// (zero-padded to avoid circular wrap-around).
pub fn analytic(x: &[f64]) -> Vec<C64> {
    let fft = Fft::new(next_fast_len(2 * x.len()));
    let mut spec = fft.forward(x);
    let last = spec.len() - 1;
    for (k, s) in spec.iter_mut().enumerate() {
        *s = if k == 0 || k == last {
            C64::new(0.0, 0.0)
        } else {
            *s * C64::new(0.0, -1.0)
        };
    }
    let h = fft.inverse(spec);
    x.iter()
        .zip(&h)
        .map(|(&re, &im)| C64::new(re, im))
        .collect()
}

/// Anti-aliasing filter of a converter, seen at an instantaneous frequency:
/// 1 up to `pass * fs`, 0 from `fs / 2` on, raised-cosine transition between.
pub fn aa_gate(f: f64, fs: f64, pass: f64) -> f64 {
    let (a, b) = (pass * fs, 0.5 * fs);
    if f <= a {
        1.0
    } else if f >= b {
        0.0
    } else {
        0.5 * (1.0 + (PI * (f - a) / (b - a)).cos())
    }
}

/// `x^n` without aliasing, for a locally sinusoidal signal (a sweep, possibly
/// filtered).
///
/// With `z = A exp(i psi)` the analytic signal of x, `x = A cos(psi)` exactly,
/// hence `x^n = A^n sum_m a_mn cos(m psi)`. Each term is a sweep of
/// instantaneous frequency `m f_inst(t)`, generated with its exact phase
/// `m psi(t)` (no resampling, hence no delay) and attenuated by the
/// anti-aliasing filter at that frequency. As long as no harmonic is cut, the
/// sum gives back `x^n` sample by sample; only the components above the
/// Nyquist frequency disappear, which an analog system followed by a
/// converter would not show either.
pub fn alias_free_power(z: &[C64], n: usize, f_inst: &[f64], fs: f64, pass: f64) -> Vec<f64> {
    let binom =
        |n: usize, k: usize| (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64);
    // cos^n(psi) = sum_m d_m cos(m psi), m = n (mod 2)
    let d: Vec<f64> = (0..=n)
        .map(|m| {
            if !(n - m).is_multiple_of(2) {
                0.0
            } else if m == 0 {
                binom(n, n / 2) / 2f64.powi(n as i32)
            } else {
                binom(n, (n - m) / 2) / 2f64.powi(n as i32 - 1)
            }
        })
        .collect();
    z.iter()
        .zip(f_inst)
        .map(|(zk, &f)| {
            let a = zk.norm();
            if a == 0.0 {
                return 0.0;
            }
            let u = zk / a;
            let mut um = C64::new(1.0, 0.0);
            let mut sum = d[0];
            for (m, &dm) in d.iter().enumerate().skip(1) {
                um *= u;
                if dm != 0.0 {
                    sum += dm * aa_gate(m as f64 * f, fs, pass) * um.re;
                }
            }
            a.powi(n as i32) * sum
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sweep::SweepSpec;

    /// Reference: FFT oversampling by R (band-limited interpolation, zero
    /// phase), power, linear anti-aliasing filter of the same shape, back to
    /// fs. Compared with `alias_free_power` below 0.4 fs.
    #[test]
    fn alias_free_power_matches_oversampling() {
        // 0.5 s and 4 s from 20 Hz to 40 kHz: the sweeps of the chain are much slower
        for (n_samples, max_db) in [(48_000, -60.0), (384_000, -80.0)] {
            check_against_oversampling(n_samples, max_db);
        }
    }

    fn check_against_oversampling(n_samples: usize, max_db: f64) {
        let fs = 96_000.0;
        let (f_start, f_end): (f64, f64) = (20.0, 40_000.0);
        let t_len = n_samples as f64 / fs;
        let l = t_len / (f_end / f_start).ln();
        let fade = |t: f64| {
            let e = 0.02;
            let w = |x: f64| (PI / 2.0 * (x / e).min(1.0)).sin().powi(2);
            w(t) * w(t_len - t)
        };
        let f_inst: Vec<f64> = (0..n_samples)
            .map(|i| f_start * (i as f64 / fs / l).exp())
            .collect();
        let x: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / fs;
                fade(t) * (2.0 * PI * f_start * l * ((t / l).exp() - 1.0)).sin()
            })
            .collect();
        let z = analytic(&x);
        let r = 32;
        let big = Fft::new(n_samples * r);
        let small = Fft::new(n_samples);
        let spec = small.forward(&x);
        let mut up = vec![C64::new(0.0, 0.0); n_samples * r / 2 + 1];
        for (k, s) in spec.iter().enumerate().take(n_samples / 2) {
            up[k] = *s * r as f64;
        }
        let x_up = big.inverse(up);
        let pass = 0.44;
        let band: Vec<usize> = (0..n_samples / 2)
            .filter(|&k| (k as f64 * fs / n_samples as f64) < 0.4 * fs)
            .collect();
        for n in [2usize, 3, 7] {
            let y_up: Vec<f64> = x_up.iter().map(|v| v.powi(n as i32)).collect();
            let s_up = big.forward(&y_up);
            let s_ref: Vec<C64> = (0..=n_samples / 2)
                .map(|k| s_up[k] * aa_gate(k as f64 * fs / n_samples as f64, fs, pass) / r as f64)
                .collect();
            let y_ref = small.inverse(s_ref);
            let y = alias_free_power(&z, n, &f_inst, fs, pass);
            let (sa, sr) = (small.forward(&y), small.forward(&y_ref));
            let num: f64 = band.iter().map(|&k| (sa[k] - sr[k]).norm_sqr()).sum();
            let den: f64 = band.iter().map(|&k| sr[k].norm_sqr()).sum();
            let err_db = 10.0 * (num / den).log10();
            // naive power (aliased), for comparison
            let naive: Vec<f64> = x.iter().map(|v| v.powi(n as i32)).collect();
            let sn = small.forward(&naive);
            let num_n: f64 = band.iter().map(|&k| (sn[k] - sr[k]).norm_sqr()).sum();
            println!(
                "T = {t_len} s, n = {n}: alias-free {err_db:7.1} dB, naive x^n {:7.1} dB",
                10.0 * (num_n / den).log10()
            );
            assert!(err_db < max_db, "T = {t_len} s, n = {n}: {err_db} dB");
        }
    }

    /// Harmonic responses of the powers of a Farina sweep whose `f_start L` is
    /// NOT an integer: `y = x^m` has a single harmonic of rank m, known
    /// exactly, `H_m = i^(1-m) / 2^(m-1)` at every frequency. The rotation of
    /// the slices by `2 pi f_start L (m - 1)` recovers its phase; without the
    /// rotation the phase would be off by that constant.
    #[test]
    fn phase_correction_of_a_non_synchronized_sweep() {
        let fs = 96_000.0;
        let sweep = Sweep::new(SweepSpec {
            fs,
            f1: 1.0,
            duration: 20.0,
            f2: fs / 2.0 * 0.6,
            margin: 1.8,
            ceiling: 0.9,
            integer_k: false,
        });
        let k = sweep.f1_sweep * sweep.l_param;
        assert!(
            (k - k.round()).abs() > 0.05,
            "k = {k} too close to an integer"
        );
        let z = analytic(&sweep.windowed);
        let n = 1 << 17;
        let w = SliceWindow {
            fade_in_ms: 100.0,
            pre_peak_ms: 400.0,
            post_peak_ms: 400.0,
            fade_out_ms: 100.0,
        };
        let deconv = Deconvolver::new(&sweep.inverse, sweep.n_samples);
        let slicer = Slicer::new(&sweep, 7, n, 0.7, &w);
        let band: Vec<usize> = (0..n / 2 + 1)
            .filter(|&b| (100.0..=10_000.0).contains(&(b as f64 * fs / n as f64)))
            .collect();
        for order in [1usize, 2, 3, 5] {
            let xn = alias_free_power(&z, order, &sweep.f_inst, fs, 0.44);
            let h = &slicer.spectra(&deconv.deconvolve(&xn))[order - 1];
            let want = C64::new(0.0, 1.0).powi(1 - order as i32) / 2f64.powi(order as i32 - 1);
            let mut worst = 0.0f64;
            for &b in &band {
                // remove the delay of the peak, at index shft of the slice
                let delay = C64::from_polar(1.0, 2.0 * PI * (b * slicer.shft) as f64 / n as f64);
                worst = worst.max((h[b] * delay / want - 1.0).norm());
            }
            println!("k = {k:.4}, H{order} of x^{order}: max relative deviation {worst:.1e}");
            assert!(worst < 1e-3, "order {order}: {worst}");
            if order > 1 {
                // the constant phase that the rotation removes is not negligible
                let phi = 2.0 * PI * k * (order - 1) as f64;
                assert!((C64::from_polar(1.0, phi) - 1.0).norm() > 0.1);
            }
        }
    }
}
