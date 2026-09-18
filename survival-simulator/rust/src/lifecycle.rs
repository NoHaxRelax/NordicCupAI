use crate::*;

#[pyfunction]
pub fn _is_position_free(
    env: &Bound<'_, PyAny>,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) -> PyResult<bool> {
    if x < 0.0 || y < 0.0 || x + width > f(env, "width")? || y + height > f(env, "height")? {
        return Ok(false);
    }
    for o in items(&env.getattr("obstacles")?)? {
        let (ox, oy) = (f(&o, "x")?, f(&o, "y")?);
        if ox < x + width
            && ox + f(&o, "width")? > x
            && oy < y + height
            && oy + f(&o, "height")? > y
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn constructor<'py>(
    env: &Bound<'py, PyAny>,
    name: &str,
    x: f64,
    y: f64,
    kw: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    // Resolve from the same module as the reference, preserving user subclass hooks.
    env.py()
        .import("src.elements.environment")?
        .getattr(name)?
        .call((x, y), kw)
}

#[pyfunction]
#[pyo3(signature=(env, x=None, y=None, parent=None))]
pub fn spawn_agent<'py>(
    env: &Bound<'py, PyAny>,
    x: Option<f64>,
    y: Option<f64>,
    parent: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = env.py();
    let rng = env.getattr("rng")?;
    let kw = PyDict::new(py);
    kw.set_item("rng", &rng)?;
    let agent;
    let mut retried = false;
    if let Some(parent) = parent {
        let angle = uniform(&rng, 0.0, 2.0 * PI)?;
        let dist = uniform(&rng, 10.0, 30.0)?;
        let mut x = (f(parent, "x")? + dist * np1(py, "cos", angle)?)
            .max(20.0)
            .min(f(env, "width")? - 20.0);
        let mut y = (f(parent, "y")? + dist * np1(py, "sin", angle)?)
            .max(20.0)
            .min(f(env, "height")? - 20.0);
        let names = [
            "speed",
            "sprint_speed",
            "max_energy",
            "hearing_radius",
            "vision_radius",
            "cone_angle",
        ];
        let caps = [
            20.0,
            40.0,
            1000.0,
            f(env, "chunk_size")? / 4.0,
            f(env, "chunk_size")?,
            PI / 2.0,
        ];
        let mut traits = [0.0; 6];
        for (i, name) in names.iter().enumerate() {
            let mut value = f(parent, name)?;
            if random(&rng)? < 0.1 {
                value *= uniform(&rng, 0.5, 1.5)?;
            }
            traits[i] = value.min(caps[i]);
        }
        // This apparently redundant constructor consumes direction and max-age RNG draws.
        let base = constructor(env, "Agent", x, y, Some(&kw))?;
        let baseline_speed = (f(&base, "speed")? + f(&base, "sprint_speed")?) / 2.0;
        let baseline_vision =
            (f(&base, "hearing_radius")? + f(&base, "vision_radius")? + f(&base, "cone_angle")?)
                / 3.0;
        let normalize = |value: f64, baseline: f64, cap: f64| {
            (128.0 + 127.0 * ((value - baseline) / cap)).clamp(0.0, 255.0) as i64
        };
        let color = (
            normalize((traits[0] + traits[1]) / 2.0, baseline_speed, 30.0),
            normalize(
                (traits[3] + traits[4] + traits[5]) / 3.0,
                baseline_vision,
                (caps[3] + caps[4] + caps[5]) / 3.0,
            ),
            normalize(traits[2], f(&base, "max_energy")?, 1000.0),
        );
        for (name, value) in names.iter().zip(traits) {
            kw.set_item(*name, value)?;
        }
        kw.set_item("color", color)?;
        if !_is_position_free(env, x, y, 20.0, 20.0)? {
            x = f(parent, "x")?;
            y = f(parent, "y")?;
        }
        agent = constructor(env, "Agent", x, y, Some(&kw))?;
    } else {
        let mut x = match x {
            Some(x) => x,
            None => uniform(&rng, 20.0, f(env, "width")? - 20.0)?,
        };
        let mut y = match y {
            Some(y) => y,
            None => uniform(&rng, 20.0, f(env, "height")? - 20.0)?,
        };
        while !_is_position_free(env, x, y, 20.0, 20.0)? {
            retried = true;
            x = uniform(&rng, 20.0, f(env, "width")? - 20.0)?;
            y = uniform(&rng, 20.0, f(env, "height")? - 20.0)?;
        }
        kw.set_item("energy", 150)?;
        agent = constructor(env, "Agent", x, y, Some(&kw))?;
    }
    let id: i64 = env.getattr("_next_agent_id")?.extract()?;
    agent.setattr("agent_id", id)?;
    env.setattr("_next_agent_id", id + 1)?;
    env.getattr("agents")?.call_method1("append", (&agent,))?;
    env.getattr("agents_dict")?.set_item(id, &agent)?;
    env.call_method0("_update_agent_grid")?;
    if retried {
        Ok(py.None().into_bound(py))
    } else {
        Ok(agent)
    }
}

#[pyfunction]
#[pyo3(signature=(env, x=None, y=None, radius=5.0, max_attempts=50))]
pub fn spawn_fruit<'py>(
    env: &Bound<'py, PyAny>,
    x: Option<f64>,
    y: Option<f64>,
    radius: f64,
    max_attempts: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let py = env.py();
    let rng = env.getattr("rng")?;
    let mut x = match x {
        Some(x) => x,
        None => uniform(&rng, 25.0, f(env, "width")? - 25.0)?,
    };
    let mut y = match y {
        Some(y) => y,
        None => uniform(&rng, 25.0, f(env, "height")? - 25.0)?,
    };
    for _ in 0..max_attempts {
        if _is_position_free(env, x, y, radius, radius)? {
            break;
        }
        x = uniform(&rng, 0.0, f(env, "width")? - radius)?;
        y = uniform(&rng, 0.0, f(env, "height")? - radius)?;
    }
    if !_is_position_free(env, x, y, radius, radius)? {
        return Ok(py.None().into_bound(py));
    }
    let kw = PyDict::new(py);
    kw.set_item("radius", radius)?;
    let fruit = constructor(env, "Fruit", x, y, Some(&kw))?;
    let id: i64 = env.getattr("_next_fruit_id")?.extract()?;
    fruit.setattr("fruit_id", id)?;
    env.setattr("_next_fruit_id", id + 1)?;
    env.getattr("fruits")?.call_method1("append", (&fruit,))?;
    env.getattr("fruits_dict")?.set_item(id, &fruit)?;
    env.call_method0("_update_fruit_grid")?;
    Ok(fruit)
}

#[pyfunction]
pub fn spawn_fruit_around_tree<'py>(
    env: &Bound<'py, PyAny>,
    tree: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let rng = env.getattr("rng")?;
    let angle = uniform(&rng, 0.0, 2.0 * PI)?;
    let dist = uniform(&rng, f(tree, "radius")?, f(tree, "radius")? * 3.0)?;
    let x = f(tree, "x")? + dist * np1(env.py(), "cos", angle)?;
    let y = f(tree, "y")? + dist * np1(env.py(), "sin", angle)?;
    spawn_fruit(env, Some(x), Some(y), 5.0, 0)
}

#[pyfunction]
#[pyo3(signature=(env, x=None, y=None, max_attempts=50))]
pub fn spawn_tree<'py>(
    env: &Bound<'py, PyAny>,
    x: Option<f64>,
    y: Option<f64>,
    max_attempts: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let py = env.py();
    let rng = env.getattr("rng")?;
    let mut x = match x {
        Some(x) => x,
        None => uniform(&rng, 25.0, f(env, "width")? - 25.0)?,
    };
    let mut y = match y {
        Some(y) => y,
        None => uniform(&rng, 25.0, f(env, "height")? - 25.0)?,
    };
    for _ in 0..max_attempts {
        if _is_position_free(env, x, y, 25.0, 25.0)? {
            break;
        }
        x = uniform(&rng, 0.0, f(env, "width")? - 25.0)?;
        y = uniform(&rng, 0.0, f(env, "height")? - 25.0)?;
    }
    // Preserve spawning even after all free-position attempts fail.
    let biome = env.getattr("biome_map")?.get_item((x as i64, y as i64))?;
    if f(&biome, "tree_spawn_rate")? > random(&rng)? {
        let tree = constructor(env, "Tree", x, y, None)?;
        env.getattr("trees")?.call_method1("append", (&tree,))?;
        env.call_method0("_update_tree_grid")?;
        Ok(tree)
    } else {
        Ok(py.None().into_bound(py))
    }
}

#[pyfunction]
#[pyo3(signature=(env, x=None, y=None, size=10.0, speed=11.0, sprint_speed=15.0, color=(255,0,0)))]
pub fn spawn_predator<'py>(
    env: &Bound<'py, PyAny>,
    x: Option<f64>,
    y: Option<f64>,
    size: f64,
    speed: f64,
    sprint_speed: f64,
    color: (i64, i64, i64),
) -> PyResult<Bound<'py, PyAny>> {
    let py = env.py();
    let rng = env.getattr("rng")?;
    let x = match x {
        Some(x) => x,
        None => uniform(&rng, 20.0, f(env, "width")? - 20.0)?,
    };
    let y = match y {
        Some(y) => y,
        None => uniform(&rng, 20.0, f(env, "height")? - 20.0)?,
    };
    if !_is_position_free(env, x, y, size, size)? {
        return Ok(py.None().into_bound(py));
    }
    let kw = PyDict::new(py);
    kw.set_item("rng", &rng)?;
    kw.set_item("size", size)?;
    kw.set_item("speed", speed)?;
    kw.set_item("sprint_speed", sprint_speed)?;
    kw.set_item("color", color)?;
    let predator = constructor(env, "Predator", x, y, Some(&kw))?;
    env.getattr("predators")?
        .call_method1("append", (&predator,))?;
    env.call_method0("_update_predator_grid")?;
    Ok(predator)
}

fn remove(
    env: &Bound<'_, PyAny>,
    object: &Bound<'_, PyAny>,
    collection: &str,
    id: Option<&str>,
    grid: &str,
) -> PyResult<()> {
    let list = env.getattr(collection)?;
    if list.contains(object)? {
        list.call_method1("remove", (object,))?;
        if let Some(id) = id {
            env.getattr(format!("{collection}_dict"))?
                .call_method1("pop", (object.getattr(id)?, env.py().None()))?;
        }
        env.call_method0(grid)?;
    }
    Ok(())
}
#[pyfunction]
pub fn kill_agent(env: &Bound<'_, PyAny>, agent: &Bound<'_, PyAny>) -> PyResult<()> {
    remove(env, agent, "agents", Some("agent_id"), "_update_agent_grid")
}
#[pyfunction]
pub fn remove_fruit(env: &Bound<'_, PyAny>, fruit: &Bound<'_, PyAny>) -> PyResult<()> {
    remove(env, fruit, "fruits", Some("fruit_id"), "_update_fruit_grid")
}
#[pyfunction]
pub fn remove_tree(env: &Bound<'_, PyAny>, tree: &Bound<'_, PyAny>) -> PyResult<()> {
    remove(env, tree, "trees", None, "_update_tree_grid")
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_is_position_free, m)?)?;
    m.add_function(wrap_pyfunction!(spawn_agent, m)?)?;
    m.add_function(wrap_pyfunction!(spawn_fruit, m)?)?;
    m.add_function(wrap_pyfunction!(spawn_fruit_around_tree, m)?)?;
    m.add_function(wrap_pyfunction!(spawn_tree, m)?)?;
    m.add_function(wrap_pyfunction!(spawn_predator, m)?)?;
    m.add_function(wrap_pyfunction!(kill_agent, m)?)?;
    m.add_function(wrap_pyfunction!(remove_fruit, m)?)?;
    m.add_function(wrap_pyfunction!(remove_tree, m)?)?;
    Ok(())
}
