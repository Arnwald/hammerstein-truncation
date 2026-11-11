//! Writes `.npy` files (NumPy format 1.0, little-endian) for the Python figures.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

fn write_header(w: &mut impl Write, descr: &str, shape: &[usize]) -> std::io::Result<()> {
    let shape_txt = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_txt}, }}");
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(header.len() as u16).to_le_bytes())?;
    w.write_all(header.as_bytes())
}

/// Array of `f64` with the given shape (row-major).
pub fn write_f64(path: &Path, shape: &[usize], data: &[f64]) {
    assert_eq!(
        shape.iter().product::<usize>(),
        data.len(),
        "{}",
        path.display()
    );
    let mut w = BufWriter::new(File::create(path).expect("create npy"));
    write_header(&mut w, "<f8", shape).unwrap();
    for v in data {
        w.write_all(&v.to_le_bytes()).unwrap();
    }
}
