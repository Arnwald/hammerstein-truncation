# Truncation error of single-level Hammerstein identification

Numerical companion of the paper

> A. Demion, "Truncation Error of Single-Level Hammerstein Identification From
> Exponential Swept Sines: Spectral Bounds, Exact Analytic Cases, and an
> Effective-Field-Theory Reading," submitted for publication.

A parallel Hammerstein cascade identified from one exponential swept-sine
measurement keeps a finite number N of harmonics, as if all orders above N
vanished. This repository computes the resulting prediction error exactly for
two analytic nonlinearities, `rho*arctan(x/rho)` and `rho*artanh(x/rho)`,
checks the closed forms against a simulated measurement chain, and produces
every figure of the paper.

## Reproducing the figures

Requirements: a Rust toolchain (1.88 or later), Python 3 with NumPy and
Matplotlib, and optionally [`just`](https://github.com/casey/just).

```sh
just figures        # build, run the three experiments, draw the figures
just test           # unit tests, clippy, rustfmt
```

Without `just`:

```sh
cargo build --release --manifest-path sim/Cargo.toml
sim/target/release/hsim exact --out output/exact
sim/target/release/hsim ess   --out output/ess
sim/target/release/hsim chain --out output/chain
python3 plots/figures.py output --out figures
```

`exact` and `ess` take a few seconds. `chain` simulates 44 measurements with a
100 s sweep at 96 kHz; it takes a few minutes and about 1.5 GB of memory
(`--duration 20` runs in seconds, with a larger deconvolution leak).

| Figure | Content | Experiment |
|---|---|---|
| `fig_ess` | deconvolved response of a ninth-order cascade to a sweep | `hsim ess` |
| `fig_radius` | singularities and Taylor polynomials of the two functions | `hsim exact` |
| `fig_bounds` | general norm bounds against the exact norms | `hsim exact` |
| `fig_curves` | exact relative error per harmonic, and its two terms | `hsim exact` |
| `fig_map` | worst harmonic over levels and radii; low-level floors | `hsim exact` |
| `fig_running` | running of the effective kernels; beta function | `hsim exact` |
| `fig_validate` | closed form against the simulated measurement chain | `hsim chain` |

The numbers quoted in the paper are printed by the experiments or checked by
the tests (for instance `exact::tests::quoted_values`,
`exact::tests::running_kernels`, `exact::tests::describing_function_beta`).

## What the code computes

All computations are in the Rust crate `sim/`; the Python script only reads
the `.npy` arrays and `meta.json` files and plots them.

- `sim/src/exact.rs`: single-level identification of `y = F(x)` at the
  reference level, prediction at any other level, the two terms of the error
  (missing orders and pollution), the general bounds, the low-level floor, and
  the running of the effective kernels with the matching scale, including the
  closed flow and the beta function of the describing function. Everything is
  exact: no sweep, no window, no noise.
- `sim/src/chain.rs`: the same error for a filtered three-branch system
  (`sim/systems/static_arctan2.0.toml`), in closed form frequency by
  frequency, and through a simulated measurement chain: Farina's sweep,
  alias-free synthesis of the system output, deconvolution, separation and
  phase rotation of the harmonic responses, removal of the deconvolution leak
  of the linear response (measured by a loopback), single-level estimation of
  seven kernels, and prediction at 43 gains from -30 to +12 dB.
- `sim/src/ess.rs`: the deconvolved response of a ninth-order cascade.
- `sim/src/statics.rs`: the static nonlinearities, their Taylor coefficients
  and closed-form harmonics, and their simulation without aliasing: the
  nonlinearity is never evaluated on the samples; its output is written
  harmonic by harmonic from the analytic signal of the sweep, each harmonic
  faded out before it reaches the Nyquist frequency.
- `sim/src/mixing.rs`: the harmonic mixing coefficients and the single-level
  estimator.
- `sim/src/sweep.rs`, `sim/src/spectral.rs`: the sweep, its inverse filter,
  the deconvolution and the separation of the harmonic responses.

### A note on the sweep of the chain check

By default the start frequency of the sweep is adjusted so that
`f_start * L` is an integer. Farina's sweep then coincides with the
synchronized sweep of Novak et al., and the constant phase
`2*pi*f_start*L*(m-1)` by which the m-th harmonic response is rotated is a
multiple of `2*pi`. `hsim chain --free-f1` keeps the start frequency as given
(`f_start * L` is then not an integer), so that the rotation is not trivial;
the unit test `spectral::tests::phase_correction_of_a_non_synchronized_sweep`
checks the rotation against harmonics known exactly.

## Layout

```
sim/            Rust crate: the experiments (binary `hsim`) and their tests
sim/systems/    the two simulated systems
plots/          figures.py
output/         written by `hsim` (not tracked)
figures/        written by plots/figures.py (not tracked)
```

## Acknowledgment

Claude Code (Anthropic), an AI assistant, was used under the author's
direction to write this code. The author verified the results and takes full
responsibility for them.

## License

MIT, see [LICENSE](LICENSE).
