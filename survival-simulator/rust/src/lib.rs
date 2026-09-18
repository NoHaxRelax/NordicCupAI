//! Rust runtime, retaining Python entity identity, RNG and public API.
//! No fast-math or fused arithmetic: operation order is part of game parity.
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use std::f64::consts::PI;
mod geometry;
mod grid;
mod lifecycle;
mod tick;
use geometry::*;
use lifecycle::*;
use tick::*;

fn f(o: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    o.getattr(name)?.extract()
}
fn items<'py>(o: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    o.try_iter()?.collect()
}
fn np1(py: Python<'_>, name: &str, a: f64) -> PyResult<f64> {
    py.import("numpy")?.getattr(name)?.call1((a,))?.extract()
}
fn np2(py: Python<'_>, name: &str, a: f64, b: f64) -> PyResult<f64> {
    py.import("numpy")?.getattr(name)?.call1((a, b))?.extract()
}
fn uniform(rng: &Bound<'_, PyAny>, a: f64, b: f64) -> PyResult<f64> {
    rng.call_method1("uniform", (a, b))?.extract()
}
fn random(rng: &Bound<'_, PyAny>) -> PyResult<f64> {
    rng.call_method0("random")?.extract()
}
fn biome<'py>(env: &Bound<'py, PyAny>, entity: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let x = (f(entity, "x")? as i64)
        .max(0)
        .min(f(env, "width")? as i64 - 1);
    let y = (f(entity, "y")? as i64)
        .max(0)
        .min(f(env, "height")? as i64 - 1);
    env.getattr("biome_map")?.get_item((x, y))
}

#[pyfunction]
#[pyo3(signature=(env, point, radius, obstacles=None))]
fn _in_obstacle(
    env: &Bound<'_, PyAny>,
    point: Point,
    radius: f64,
    obstacles: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let all = env.getattr("obstacles")?;
    Ok(blocked(
        point,
        radius,
        &rectangles(obstacles.unwrap_or(&all))?,
    ))
}

#[pyfunction]
fn update_entity_energy(
    entity_env: &Bound<'_, PyAny>,
    entity: &Bound<'_, PyAny>,
    energy_cost: f64,
) -> PyResult<()> {
    let _ = entity_env;
    entity.setattr("energy", f(entity, "energy")? - energy_cost)
}

#[pyfunction]
fn update_entity_direction(
    env: &Bound<'_, PyAny>,
    entity: &Bound<'_, PyAny>,
    turn_angle: f64,
) -> PyResult<()> {
    entity.setattr("direction", f(entity, "direction")? + turn_angle)?;
    update_entity_energy(env, entity, PI.min(turn_angle.abs()) / (2.0 * PI))
}

#[pyfunction]
fn _keep_agent_in_bounds(env: &Bound<'_, PyAny>, agent: &Bound<'_, PyAny>) -> PyResult<()> {
    let size = f(agent, "size")?;
    agent.setattr("x", size.max((f(env, "width")? - size).min(f(agent, "x")?)))?;
    agent.setattr(
        "y",
        size.max((f(env, "height")? - size).min(f(agent, "y")?)),
    )
}

#[pyfunction]
#[pyo3(signature=(env, entity, distance, direction=None, local_obstacles=None))]
fn update_entity_position(
    env: &Bound<'_, PyAny>,
    entity: &Bound<'_, PyAny>,
    mut distance: f64,
    direction: Option<f64>,
    local_obstacles: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let py = env.py();
    let (speed, sprint) = (f(entity, "speed")?, f(entity, "sprint_speed")?);
    if distance < 0.0 {
        distance = 0.0;
    }
    if distance > sprint {
        distance = sprint;
    }
    if f(entity, "energy")? < f(entity, "max_energy")? / 5.0 && distance > speed {
        distance = speed;
    }
    let cost = if distance <= speed {
        distance * 0.05
    } else {
        speed * 0.05 + (distance - speed) * 0.5
    };
    update_entity_energy(env, entity, cost)?;
    let dir = match direction {
        Some(d) => f(entity, "direction")? + d,
        None => f(entity, "direction")?,
    };
    distance *= f(&biome(env, entity)?, "move_penalty")?;
    let (x, y, size) = (f(entity, "x")?, f(entity, "y")?, f(entity, "size")?);
    let all = env.getattr("obstacles")?;
    let rects = rectangles(local_obstacles.unwrap_or(&all))?;
    let p = (
        x + distance * np1(py, "cos", dir)?,
        y + distance * np1(py, "sin", dir)?,
    );
    if blocked(p, size, &rects) {
        let step = PI / 18.0;
        for i in 0..(2.0 * PI / step) as usize {
            let angle = dir + step * i.div_ceil(2) as f64 * if i % 2 == 0 { 1.0 } else { -1.0 };
            let p = (
                x + distance * np1(py, "cos", angle)?,
                y + distance * np1(py, "sin", angle)?,
            );
            if !blocked(p, size, &rects) {
                entity.setattr("x", p.0)?;
                entity.setattr("y", p.1)?;
                break;
            }
        }
    } else {
        entity.setattr("x", p.0)?;
        entity.setattr("y", p.1)?;
    }
    _keep_agent_in_bounds(env, entity)
}

#[pyfunction]
#[pyo3(signature=(env, agent_id, move_distance, move_direction, turn_angle, spawn_agent=false))]
fn agent_step(
    env: &Bound<'_, PyAny>,
    agent_id: i64,
    move_distance: f64,
    move_direction: Option<f64>,
    turn_angle: f64,
    spawn_agent: bool,
) -> PyResult<()> {
    let agent = env
        .getattr("agents_dict")?
        .call_method1("get", (agent_id,))?;
    if agent.is_none() {
        return Ok(());
    }
    let obstacles = env.call_method1("_get_local_obstacles", (&agent,))?;
    update_entity_position(env, &agent, move_distance, move_direction, Some(&obstacles))?;
    env.call_method0("_update_agent_grid")?;
    update_entity_direction(env, &agent, turn_angle)?;
    if spawn_agent && f(&agent, "energy")? > 100.0 {
        lifecycle::spawn_agent(env, None, None, Some(&agent))?;
        agent.setattr("energy", f(&agent, "energy")? - 100.0)?;
    }
    Ok(())
}

#[pyfunction]
fn get_agent_state<'py>(env: &Bound<'py, PyAny>, agent_id: i64) -> PyResult<Bound<'py, PyAny>> {
    let py = env.py();
    let agent = env
        .getattr("agents_dict")?
        .call_method1("get", (agent_id,))?;
    if agent.is_none() {
        return Ok(py.None().into_bound(py));
    }
    let out = PyDict::new(py);
    out.set_item("agent_id", agent_id)?;
    out.set_item(
        "observations",
        env.getattr("agent_observations")?
            .call_method1("get", (agent_id, PyList::empty(py)))?,
    )?;
    out.set_item("biome", biome(env, &agent)?.getattr("type")?)?;
    for (key, attr) in [
        ("energy", "energy"),
        ("age", "age"),
        ("speed", "speed"),
        ("sprint_speed", "sprint_speed"),
        ("hearing_radius", "hearing_radius"),
        ("vision_angle", "cone_angle"),
        ("vision_range", "vision_radius"),
        ("max_energy", "max_energy"),
    ] {
        out.set_item(key, agent.getattr(attr)?)?;
    }
    Ok(out.into_any())
}

#[pymodule]
fn _survival_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_visibility, m)?)?;
    m.add_function(wrap_pyfunction!(observe, m)?)?;
    m.add_function(wrap_pyfunction!(_in_obstacle, m)?)?;
    m.add_function(wrap_pyfunction!(_keep_agent_in_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(update_entity_position, m)?)?;
    m.add_function(wrap_pyfunction!(update_entity_direction, m)?)?;
    m.add_function(wrap_pyfunction!(update_entity_energy, m)?)?;
    m.add_function(wrap_pyfunction!(agent_step, m)?)?;
    m.add_function(wrap_pyfunction!(non_agent_step, m)?)?;
    m.add_function(wrap_pyfunction!(get_agent_state, m)?)?;
    lifecycle::register(m)?;
    grid::register(m)?;
    Ok(())
}
