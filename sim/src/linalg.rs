//! Singular values of a small real matrix.

/// Singular values of a real matrix (one-sided Jacobi SVD).
pub fn singular_values(a: &[Vec<f64>]) -> Vec<f64> {
    let rows = a.len();
    let cols = a[0].len();
    let mut u: Vec<Vec<f64>> = (0..cols)
        .map(|j| (0..rows).map(|i| a[i][j]).collect())
        .collect();
    for _ in 0..100 {
        let mut rotated = false;
        for p in 0..cols {
            for q in p + 1..cols {
                let alpha: f64 = u[p].iter().map(|x| x * x).sum();
                let beta: f64 = u[q].iter().map(|x| x * x).sum();
                let gamma: f64 = u[p].iter().zip(&u[q]).map(|(x, y)| x * y).sum();
                if gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..rows {
                    let (x, y) = (u[p][i], u[q][i]);
                    u[p][i] = c * x - s * y;
                    u[q][i] = s * x + c * y;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    u.iter()
        .map(|c| c.iter().map(|x| x * x).sum::<f64>().sqrt())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singular_values_of_a_known_matrix() {
        // [[3, 0], [4, 5]]: singular values sqrt(45) and sqrt(5)
        let mut sv = singular_values(&[vec![3.0, 0.0], vec![4.0, 5.0]]);
        sv.sort_by(|a, b| b.total_cmp(a));
        assert!((sv[0] - 45f64.sqrt()).abs() < 1e-14);
        assert!((sv[1] - 5f64.sqrt()).abs() < 1e-14);
    }
}
