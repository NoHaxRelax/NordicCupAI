use pyo3::prelude::*;
mod pyrandom;

/// Python-visible wrapper around the bit-exact CPython RNG port (for differential tests).
#[pyclass]
struct PyRandom { inner: pyrandom::PyRandom }

#[pymethods]
impl PyRandom {
    #[new]
    fn new(seed: u64) -> Self { PyRandom { inner: pyrandom::PyRandom::new(seed) } }
    fn random(&mut self) -> f64 { self.inner.random() }
    fn uniform(&mut self, a: f64, b: f64) -> f64 { self.inner.uniform(a, b) }
    fn getrandbits(&mut self, k: u32) -> u64 { self.inner.getrandbits(k) }
    fn randint(&mut self, a: i64, b: i64) -> i64 { self.inner.randint(a, b) }
    fn choice_index(&mut self, n: usize) -> usize { self.inner.choice_index(n) }
    fn genrand_u32(&mut self) -> u32 { self.inner.genrand_u32() }
}

#[pymodule]
fn rustsim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRandom>()?;
    Ok(())
}
