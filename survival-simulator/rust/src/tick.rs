use crate::*;
use pyo3::types::PyFloat;

fn pow(py: Python<'_>, a: f64, b: f64) -> PyResult<f64> {
    PyFloat::new(py, a).call_method1("__pow__", (b,))?.extract()
}

// (distance, relative movement direction, optional turn), matching Predator.step.
fn predator_signals(
    predator: &Bound<'_, PyAny>,
    obs: &Bound<'_, PyList>,
) -> PyResult<(f64, Option<f64>, Option<f64>)> {
    let py = predator.py();
    let mut closest: Option<Bound<'_, PyAny>> = None;
    let mut distance = f64::INFINITY;
    let mut edges: Vec<Edge> = Vec::new();
    for o in obs.iter() {
        let tag: String = o.get_item("type")?.extract()?;
        if tag == "Agent" {
            let d: f64 = o.get_item("distance")?.extract()?;
            if d < distance {
                distance = d;
                closest = Some(o);
            }
        } else if tag == "Edge" {
            edges.push(o.get_item("coords")?.extract()?);
        }
    }
    let sprint = f(predator, "sprint_speed")?;
    if let Some(agent) = closest {
        let angle: f64 = agent.get_item("angle")?.extract()?;
        let rel: f64 = agent.get_item("rel_dir")?.extract()?;
        if rel.abs() > PI * 1.0 / 2.0 || distance < f(predator, "hearing_radius")? * 1.5 {
            let turn = (-0.3_f64).max(0.3_f64.min(angle * 0.5));
            return if angle.abs() > 0.05 {
                Ok((sprint.min(distance), Some(turn), Some(turn)))
            } else {
                Ok((sprint.min(distance), Some(angle), None))
            };
        }
        let sign = if rel > 0.0 {
            -1.0
        } else if rel < 0.0 {
            1.0
        } else {
            -0.0
        };
        let dir = angle + sign * PI * 1.0 / 4.0;
        let dx = sprint * np1(py, "cos", dir)?;
        let dy = sprint * np1(py, "sin", dir)?;
        let x = distance * np1(py, "cos", angle)? - dx;
        let y = distance * np1(py, "sin", angle)? - dy;
        return Ok((sprint, Some(dir), Some(np2(py, "arctan2", y, x)?)));
    }
    if !edges.is_empty() {
        let mut best = (0.0, 0.0);
        let mut dist = f64::INFINITY;
        for ((x1, y1), (x2, y2)) in edges {
            let (dx, dy) = (x2 - x1, y2 - y1);
            let t = (-(x1 * dx + y1 * dy)) / (dx * dx + dy * dy);
            // Python max(0,min(1,nan)) returns 1, as does f64::min here.
            let t = 0.0_f64.max(1.0_f64.min(t));
            let p = (x1 + t * dx, y1 + t * dy);
            let d = np2(py, "hypot", p.0, p.1)?;
            if d < dist {
                dist = d;
                best = p;
            }
        }
        let dist = (dist - f(predator, "size")?).max(2.0);
        let angle = wrap(np2(py, "arctan2", best.1, best.0)?);
        let turn = if angle > 0.0 { -PI / dist } else { PI / dist };
        return Ok((f(predator, "speed")?, Some(turn), Some(turn)));
    }
    let turn = uniform(&predator.getattr("rng")?, -0.1, 0.1)?;
    Ok((f(predator, "speed")?, None, Some(turn)))
}

#[pyfunction]
pub fn non_agent_step(env: &Bound<'_, PyAny>, dt: f64) -> PyResult<()> {
    let py = env.py();
    let rng = env.getattr("rng")?;
    // Use live list indexing: removal deliberately skips the next entity,
    // exactly as the original Python for-loop does.
    let agents = env.getattr("agents")?;
    let mut i = 0;
    while i < agents.len()? {
        let agent = agents.get_item(i)?;
        i += 1;
        agent.setattr("age", f(&agent, "age")? + dt)?;
        agent.setattr(
            "energy",
            f(&agent, "energy")? - dt * f(&biome(env, &agent)?, "energy_drain_rate")?,
        )?;
        if f(&agent, "energy")? <= 0.0 {
            kill_agent(env, &agent)?;
            continue;
        }
        if f(&agent, "age")? > f(&agent, "max_age")? {
            agent.setattr("energy", f(&agent, "energy")? - 0.01 * f(&agent, "age")?)?;
        }
        let local = env.call_method1("_get_local_objects", (&agent,))?;
        let a = local.get_item(0)?;
        let fruit = local.get_item(1)?;
        let trees = local.get_item(2)?;
        let pred = local.get_item(4)?;
        let edges = local.get_item(5)?;
        let obs = observe(&agent, &a, Some(&fruit), Some(&trees), Some(&pred), &edges)?;
        env.getattr("agent_observations")?
            .set_item(agent.getattr("agent_id")?, obs)?;
        for fruit in items(&fruit)?.iter().rev() {
            let d = np2(
                py,
                "hypot",
                f(&agent, "x")? - f(fruit, "x")?,
                f(&agent, "y")? - f(fruit, "y")?,
            )?;
            if d < f(&agent, "size")? + f(fruit, "radius")? {
                agent.setattr(
                    "energy",
                    f(&agent, "max_energy")?.min(f(&agent, "energy")? + f(fruit, "energy")?),
                )?;
                env.setattr("score", f(env, "score")? + f(fruit, "energy")? / 1000.0)?;
                remove_fruit(env, fruit)?;
            }
        }
    }
    for predator in items(&env.getattr("predators")?)? {
        if predator.getattr("resting")?.is_truthy()? {
            if f(&predator, "energy")? > f(&predator, "max_energy")? * 0.5 {
                predator.setattr("resting", false)?;
            } else {
                predator.setattr("energy", f(&predator, "energy")? + dt * 30.0)?;
                continue;
            }
        }
        let agents = env.call_method1("_get_local_agents", (&predator,))?;
        let edges = env.call_method1("_get_local_edges", (&predator,))?;
        let obstacles = env.call_method1("_get_local_obstacles", (&predator,))?;
        let obs = observe(&predator, &agents, None, None, None, &edges)?;
        let (distance, direction, turn) = predator_signals(&predator, &obs)?;
        update_entity_position(env, &predator, distance, direction, Some(&obstacles))?;
        env.call_method0("_update_predator_grid")?;
        if let Some(turn) = turn {
            update_entity_direction(env, &predator, turn)?;
        }
        let local = items(&env.call_method1("_get_local_agents", (&predator,))?)?;
        // Preserve NumPy's vector hypot, including the reversed touching order.
        if !local.is_empty() {
            let dx = local
                .iter()
                .map(|a| Ok(f(a, "x")? - f(&predator, "x")?))
                .collect::<PyResult<Vec<_>>>()?;
            let dy = local
                .iter()
                .map(|a| Ok(f(a, "y")? - f(&predator, "y")?))
                .collect::<PyResult<Vec<_>>>()?;
            let distances: Vec<f64> = py
                .import("numpy")?
                .getattr("hypot")?
                .call1((dx, dy))?
                .extract()?;
            for (agent, d) in local.iter().zip(distances).rev() {
                if d < f(&predator, "size")? + f(agent, "size")? {
                    predator.setattr(
                        "energy",
                        f(&predator, "max_energy")?
                            .min(f(&predator, "energy")? + f(agent, "energy")?),
                    )?;
                    env.setattr("score", f(env, "score")? - f(agent, "energy")? / 100.0)?;
                    kill_agent(env, agent)?;
                }
            }
        }
        if f(&predator, "energy")? <= 0.0 {
            predator.setattr("resting", true)?;
        }
    }
    let fruits = env.getattr("fruits")?;
    let mut i = 0;
    while i < fruits.len()? {
        let fruit = fruits.get_item(i)?;
        i += 1;
        if f(&fruit, "age")? > 100.0 {
            remove_fruit(env, &fruit)?;
            continue;
        }
        let amount = 2.0 * dt;
        fruit.setattr("age", f(&fruit, "age")? + amount)?;
        if f(&fruit, "energy")? < 60.0 {
            fruit.setattr("energy", f(&fruit, "energy")? + amount)?;
            fruit.setattr("radius", f(&fruit, "radius")? + amount * 0.1)?;
        } else {
            let (r, g, _): (f64, f64, f64) = fruit.getattr("color")?.extract()?;
            fruit.setattr(
                "color",
                (
                    150.0_f64.min(r + amount * 255.0 / 60.0),
                    100.0_f64.max(g - amount * 255.0 / 60.0),
                    0,
                ),
            )?;
        }
    }
    let trees = env.getattr("trees")?;
    let mut chance = 100.0 / (trees.len()? as f64 / 2.0).max(1.0) * dt;
    chance *= pow(py, 0.5, f(env, "time")? / 300.0)?;
    if random(&rng)? < chance {
        spawn_tree(env, None, None, 50)?;
    }
    let mut i = 0;
    while i < trees.len()? {
        let tree = trees.get_item(i)?;
        i += 1;
        tree.setattr("radius", 20.0_f64.min(f(&tree, "radius")? + dt))?;
        tree.setattr("age", f(&tree, "age")? + dt)?;
        if f(&tree, "age")? > 50.0 + 50.0 * pow(py, random(&rng)?, 0.5)? {
            remove_tree(env, &tree)?;
            continue;
        } else if f(&tree, "age")? >= 20.0
            && random(&rng)? < dt * f(&biome(env, &tree)?, "fruit_spawn_rate")?
        {
            spawn_fruit_around_tree(env, &tree)?;
        }
    }
    env.setattr("time", f(env, "time")? + dt)?;
    env.setattr("score", f(env, "score")? + dt)?;
    let chance =
        (1.0 / (env.getattr("predators")?.len()? as f64).max(1.0)) * dt * f(env, "time")? * 0.0001;
    if chance > random(&rng)? {
        spawn_predator(env, None, None, 10.0, 11.0, 15.0, (255, 0, 0))?;
    }
    Ok(())
}
