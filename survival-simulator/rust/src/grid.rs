//! Keep CPython sets and their insertion/union semantics: their iteration order
//! affects tie-breaking, observation order, fruit accumulation and predator kills.
use crate::*;
use pyo3::types::{PySet, PyTuple};

fn chunk(env: &Bound<'_, PyAny>, x: f64, y: f64) -> PyResult<(i64, i64)> {
    // Use Python floor division, including its rounding behavior at cell borders.
    env.call_method1("to_chunk", (x, y))?.extract()
}

fn update(env: &Bound<'_, PyAny>, collection: &str, grid_name: &str) -> PyResult<()> {
    let py = env.py();
    let grid = py
        .import("collections")?
        .getattr("defaultdict")?
        .call1((py.get_type::<PySet>(),))?;
    for obj in items(&env.getattr(collection)?)? {
        let (x, y) = (f(&obj, "x")?, f(&obj, "y")?);
        let cell = chunk(env, x, y)?;
        grid.get_item(cell)?.cast::<PySet>()?.add(obj)?;
    }
    env.setattr(grid_name, grid)
}

macro_rules! update_grid {
    ($name:ident,$collection:literal,$grid:literal) => {
        #[pyfunction]
        pub fn $name(env: &Bound<'_, PyAny>) -> PyResult<()> {
            update(env, $collection, $grid)
        }
    };
}
update_grid!(_update_agent_grid, "agents", "grid_agents");
update_grid!(_update_fruit_grid, "fruits", "grid_fruits");
update_grid!(_update_tree_grid, "trees", "grid_trees");
update_grid!(_update_predator_grid, "predators", "grid_predators");

fn local<'py>(
    env: &Bound<'py, PyAny>,
    creature: &Bound<'py, PyAny>,
    grid_name: &str,
    remove_self: bool,
) -> PyResult<Bound<'py, PySet>> {
    let out = PySet::empty(env.py())?;
    let grid = env.getattr(grid_name)?;
    let grid = grid.cast::<PyDict>()?;
    let (cx, cy) = chunk(env, f(creature, "x")?, f(creature, "y")?)?;
    for dx in -1..=1 {
        for dy in -1..=1 {
            if let Some(objects) = grid.get_item((cx + dx, cy + dy))? {
                out.call_method1("update", (objects,))?;
            }
        }
    }
    if remove_self {
        out.discard(creature)?;
    }
    Ok(out)
}

macro_rules! local_grid {
    ($name:ident,$grid:literal,$remove:literal) => {
        #[pyfunction]
        pub fn $name<'py>(
            env: &Bound<'py, PyAny>,
            creature: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PySet>> {
            local(env, creature, $grid, $remove)
        }
    };
}
local_grid!(_get_local_agents, "grid_agents", true);
local_grid!(_get_local_fruits, "grid_fruits", false);
local_grid!(_get_local_trees, "grid_trees", false);
local_grid!(_get_local_obstacles, "grid_obstacles", false);
local_grid!(_get_local_predators, "grid_predators", true);
local_grid!(_get_local_edges, "grid_edges", false);

#[pyfunction]
pub fn _get_local_objects<'py>(
    env: &Bound<'py, PyAny>,
    creature: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut grids = Vec::new();
    let mut sets = Vec::new();
    for name in [
        "agents",
        "fruits",
        "trees",
        "obstacles",
        "predators",
        "edges",
    ] {
        grids.push(env.getattr(format!("grid_{name}"))?);
        sets.push(PySet::empty(env.py())?);
    }
    let (cx, cy) = chunk(env, f(creature, "x")?, f(creature, "y")?)?;
    for dx in -1..=1 {
        for dy in -1..=1 {
            for (grid, set) in grids.iter().zip(&sets) {
                if let Some(objects) = grid.cast::<PyDict>()?.get_item((cx + dx, cy + dy))? {
                    set.call_method1("update", (objects,))?;
                }
            }
        }
    }
    if sets[0].contains(creature)? {
        sets[0].discard(creature)?;
    } else if sets[4].contains(creature)? {
        sets[4].discard(creature)?;
    }
    PyTuple::new(env.py(), sets)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_update_agent_grid, m)?)?;
    m.add_function(wrap_pyfunction!(_update_fruit_grid, m)?)?;
    m.add_function(wrap_pyfunction!(_update_tree_grid, m)?)?;
    m.add_function(wrap_pyfunction!(_update_predator_grid, m)?)?;
    m.add_function(wrap_pyfunction!(_get_local_agents, m)?)?;
    m.add_function(wrap_pyfunction!(_get_local_fruits, m)?)?;
    m.add_function(wrap_pyfunction!(_get_local_trees, m)?)?;
    m.add_function(wrap_pyfunction!(_get_local_obstacles, m)?)?;
    m.add_function(wrap_pyfunction!(_get_local_predators, m)?)?;
    m.add_function(wrap_pyfunction!(_get_local_edges, m)?)?;
    m.add_function(wrap_pyfunction!(_get_local_objects, m)?)?;
    Ok(())
}
