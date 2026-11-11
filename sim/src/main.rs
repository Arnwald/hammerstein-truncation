//! `hsim EXPERIMENT [options]`: runs one of the numerical experiments of the paper.

use hammerstein_truncation::{chain, ess, exact};

const USAGE: &str = "usage: hsim EXPERIMENT [options]

experiments:
  exact [--out DIR]
      exact single-level truncation error for rho*arctan(x/rho) and
      rho*artanh(x/rho): curves, bounds, floors, map, running of the kernels
  chain [--out DIR] [--duration S] [--free-f1]
      prediction error of a filtered test system, in closed form and through a
      simulated swept-sine measurement chain (default: 100 s sweep, about
      1.5 GB of memory); --free-f1 keeps the start frequency of the sweep as
      given instead of adjusting it so that f_start*L is an integer
  ess [--out DIR]
      deconvolved response of a ninth-order cascade to a 10 s sweep

Outputs go to output/<experiment> at the root of the repository by default.";

fn main() {
    let mut argv = std::env::args().skip(1);
    let Some(experiment) = argv.next() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    let rest: Vec<String> = argv.collect();
    match experiment.as_str() {
        "exact" => exact::run(rest),
        "chain" => chain::run(rest),
        "ess" => ess::run(rest),
        "-h" | "--help" | "help" => println!("{USAGE}"),
        other => {
            eprintln!("unknown experiment: {other}\n\n{USAGE}");
            std::process::exit(2);
        }
    }
}
