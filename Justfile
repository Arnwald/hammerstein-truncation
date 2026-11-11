# Numerical companion of "Truncation Error of Single-Level Hammerstein
# Identification From Exponential Swept Sines".

hsim := "sim/target/release/hsim"

default: figures

# Build the simulator
build:
	cargo build --release --manifest-path sim/Cargo.toml

# Tests, lints and formatting
test:
	cargo test --release --manifest-path sim/Cargo.toml
	cargo clippy --release --manifest-path sim/Cargo.toml --all-targets -- -D warnings
	cargo fmt --manifest-path sim/Cargo.toml --check

# Run the three experiments (the chain takes a few minutes and about 1.5 GB)
data: build
	{{hsim}} exact --out output/exact
	{{hsim}} ess --out output/ess
	{{hsim}} chain --out output/chain

# All the figures of the paper, as PDF, in figures/
figures: data
	python3 plots/figures.py output --out figures
	rm -f figures/*.png

# The chain check with a sweep whose start frequency is not adjusted
# (f_start*L not an integer: the phase rotation of the harmonics is not trivial)
chain-free-f1: build
	{{hsim}} chain --free-f1 --out output/chain_free_f1
