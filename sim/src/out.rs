//! Outputs of an experiment: a directory, `.npy` arrays and a `meta.json`.

use std::fmt::Write as _;
use std::path::PathBuf;

use crate::npy;

/// Output directory of an experiment.
pub struct Out {
    pub dir: PathBuf,
}

impl Out {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|e| panic!("output directory {}: {e}", dir.display()));
        Out { dir }
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    pub fn f64(&self, name: &str, shape: &[usize], data: &[f64]) {
        npy::write_f64(&self.path(name), shape, data);
    }

    /// Rows of equal length, stacked.
    pub fn f64_rows(&self, name: &str, rows: &[Vec<f64>]) {
        let width = rows.first().map_or(0, Vec::len);
        assert!(
            rows.iter().all(|r| r.len() == width),
            "{name}: rows of unequal length"
        );
        let flat: Vec<f64> = rows.concat();
        self.f64(name, &[rows.len(), width], &flat);
    }

    pub fn text(&self, name: &str, text: &str) {
        let path = self.path(name);
        std::fs::write(&path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }
}

/// A flat JSON object built key by key.
#[derive(Default)]
pub struct Json {
    fields: Vec<(String, String)>,
}

fn num(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".into()
    }
}

fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl Json {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(mut self, key: &str, value: String) -> Self {
        self.fields.push((key.into(), value));
        self
    }

    pub fn num(self, key: &str, v: f64) -> Self {
        self.push(key, num(v))
    }

    pub fn int(self, key: &str, v: i64) -> Self {
        self.push(key, v.to_string())
    }

    pub fn bool(self, key: &str, v: bool) -> Self {
        self.push(key, v.to_string())
    }

    pub fn str(self, key: &str, v: &str) -> Self {
        self.push(key, quote(v))
    }

    pub fn nums(self, key: &str, v: &[f64]) -> Self {
        let items: Vec<String> = v.iter().map(|&x| num(x)).collect();
        self.push(key, format!("[{}]", items.join(", ")))
    }

    pub fn strs(self, key: &str, v: &[&str]) -> Self {
        let items: Vec<String> = v.iter().map(|s| quote(s)).collect();
        self.push(key, format!("[{}]", items.join(", ")))
    }

    pub fn write(&self, out: &Out, name: &str) {
        let mut s = String::from("{\n");
        for (i, (k, v)) in self.fields.iter().enumerate() {
            let sep = if i + 1 < self.fields.len() { "," } else { "" };
            let _ = writeln!(s, "  {}: {v}{sep}", quote(k));
        }
        s.push_str("}\n");
        out.text(name, &s);
    }
}
