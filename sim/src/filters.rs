//! Biquad sections (b0 b1 b2 a0 a1 a2, a0 = 1) and cascaded filtering.

use std::f64::consts::PI;

pub type Sos = [f64; 6];

fn normalize(b: [f64; 3], a: [f64; 3]) -> Sos {
    [
        b[0] / a[0],
        b[1] / a[0],
        b[2] / a[0],
        1.0,
        a[1] / a[0],
        a[2] / a[0],
    ]
}

/// Low-pass or high-pass biquad of the RBJ cookbook (bilinear transform
/// pre-warped at fc).
fn rbj(fc: f64, q: f64, fs: f64, highpass: bool) -> Sos {
    let w0 = 2.0 * PI * fc / fs;
    let cw = w0.cos();
    let alpha = w0.sin() / (2.0 * q);
    let b = if highpass {
        [(1.0 + cw) / 2.0, -(1.0 + cw), (1.0 + cw) / 2.0]
    } else {
        [(1.0 - cw) / 2.0, 1.0 - cw, (1.0 - cw) / 2.0]
    };
    normalize(b, [1.0 + alpha, -2.0 * cw, 1.0 - alpha])
}

/// Butterworth filter of even order: a cascade of RBJ biquads with the Q
/// factors of the prototype. Same transfer function as `scipy.signal.butter`
/// (pre-warped bilinear transform).
pub fn butterworth(order: usize, fc: f64, fs: f64, highpass: bool) -> Vec<Sos> {
    assert!(order.is_multiple_of(2), "even order only");
    (1..=order / 2)
        .map(|k| {
            let q = 1.0 / (2.0 * ((2 * k - 1) as f64 * PI / (2 * order) as f64).sin());
            rbj(fc, q, fs, highpass)
        })
        .collect()
}

/// Peaking equalizer (RBJ cookbook): centre frequency, Q, gain in dB.
pub fn peq(fc: f64, q: f64, gain_db: f64, fs: f64) -> Sos {
    let a = 10f64.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * fc / fs;
    let alpha = w0.sin() / (2.0 * q);
    normalize(
        [1.0 + alpha * a, -2.0 * w0.cos(), 1.0 - alpha * a],
        [1.0 + alpha / a, -2.0 * w0.cos(), 1.0 - alpha / a],
    )
}

/// In-place filtering by a cascade of biquads (transposed direct form II).
pub fn sosfilt(chain: &[Sos], x: &mut [f64]) {
    for s in chain {
        let (mut z1, mut z2) = (0.0, 0.0);
        for v in x.iter_mut() {
            let input = *v;
            let y = s[0] * input + z1;
            z1 = s[1] * input - s[4] * y + z2;
            z2 = s[2] * input - s[5] * y;
            *v = y;
        }
    }
}
