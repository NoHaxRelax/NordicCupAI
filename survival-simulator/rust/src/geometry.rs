//! Native ray intersections and sensing. NumPy supplies the reference ufuncs;
//! arithmetic, filtering, ordering, and observation assembly run in Rust.
use crate::*;

pub type Point = (f64, f64);
pub type Edge = (Point, Point);

pub fn wrap(a: f64) -> f64 {
    (a + PI).rem_euclid(2.0 * PI) - PI
}

fn unary(py: Python<'_>, name: &str, values: Vec<f64>) -> PyResult<Vec<f64>> {
    py.import("numpy")?
        .getattr(name)?
        .call1((values,))?
        .extract()
}

fn binary(py: Python<'_>, name: &str, a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    py.import("numpy")?.getattr(name)?.call1((a, b))?.extract()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)] // Match the public Python signature.
#[pyo3(signature = (x, y, direction, cone_angle, vision_radius, edges, epsilon=1e-3))]
pub fn compute_visibility(
    py: Python<'_>,
    x: f64,
    y: f64,
    direction: f64,
    cone_angle: f64,
    vision_radius: f64,
    edges: Vec<Edge>,
    epsilon: f64,
) -> PyResult<(Vec<Point>, Vec<Edge>)> {
    // The oracle indexes corners[:, 0] before its intended empty-edge branch.
    // Preserve that error instead of silently changing invalid-world behavior.
    if edges.is_empty() {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            "too many indices for array: array is 1-dimensional, but 2 were indexed",
        ));
    }
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for &(a, b) in &edges {
        for (px, py) in [a, b] {
            let (vx, vy) = (px - x, py - y);
            if vx * vx + vy * vy <= vision_radius * vision_radius {
                dx.push(vx);
                dy.push(vy);
            }
        }
    }
    let mut rays = Vec::new();
    if !dx.is_empty() {
        for angle in binary(py, "arctan2", dy, dx)? {
            for offset in [-epsilon, 0.0, epsilon] {
                let a = angle + offset;
                if wrap(a - direction).abs() <= cone_angle / 2.0 {
                    rays.push(a);
                }
            }
        }
    }
    rays.extend([
        direction - cone_angle / 2.0,
        direction + cone_angle / 2.0,
        direction - cone_angle / 4.0,
        direction + cone_angle / 4.0,
        direction,
    ]);
    rays.sort_by(f64::total_cmp);
    rays.dedup_by(|a, b| *a == *b);
    let cos = unary(py, "cos", rays.clone())?;
    let sin = unary(py, "sin", rays)?;
    let mut points = Vec::with_capacity(cos.len());
    let mut hits = Vec::new();
    for (rx, ry) in cos.into_iter().zip(sin) {
        let mut closest = f64::INFINITY;
        let mut index = 0;
        for (j, &((x1, y1), (x2, y2))) in edges.iter().enumerate() {
            let (vx, vy) = (x2 - x1, y2 - y1);
            let det = -rx * vy + ry * vx;
            if det.abs() < 1e-8 {
                continue;
            }
            let t = (-vy * (x1 - x) + vx * (y1 - y)) / det;
            let u = (-ry * (x1 - x) + rx * (y1 - y)) / det;
            if t >= 0.0 && (0.0..=1.0).contains(&u) && t < closest {
                closest = t;
                index = j;
            }
        }
        let t = closest.min(vision_radius);
        points.push((x + rx * t, y + ry * t));
        if t < vision_radius {
            hits.push(edges[index]);
        }
    }
    if !edges.is_empty() {
        let angles = binary(
            py,
            "arctan2",
            points.iter().map(|p| p.1 - y).collect(),
            points.iter().map(|p| p.0 - x).collect(),
        )?;
        let mut keyed: Vec<_> = points.into_iter().zip(angles).collect();
        keyed.sort_by(|a, b| wrap(a.1 - direction).total_cmp(&wrap(b.1 - direction)));
        points = keyed.into_iter().map(|p| p.0).collect();
    }
    Ok((points, hits))
}

#[pyfunction]
#[pyo3(signature=(creature, agents, fruits, trees, predators, edges))]
pub fn observe<'py>(
    creature: &Bound<'py, PyAny>,
    agents: &Bound<'py, PyAny>,
    fruits: Option<&Bound<'py, PyAny>>,
    trees: Option<&Bound<'py, PyAny>>,
    predators: Option<&Bound<'py, PyAny>>,
    edges: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let py = creature.py();
    let (x, y, dir) = (
        f(creature, "x")?,
        f(creature, "y")?,
        f(creature, "direction")?,
    );
    let (hearing, vision, cone) = (
        f(creature, "hearing_radius")?,
        f(creature, "vision_radius")?,
        f(creature, "cone_angle")?,
    );
    let edge_vec = items(edges)?
        .iter()
        .map(|e| e.extract())
        .collect::<PyResult<Vec<Edge>>>()?;
    let (poly, hits) = compute_visibility(py, x, y, dir, cone, vision, edge_vec, 1e-3)?;
    creature.setattr("_vision_poly", &poly)?;
    let out = PyList::empty(py);
    let mut shape: Option<Bound<'py, PyAny>> = None;
    for (objects, tag, include_direction, include_id) in [
        (fruits, "Fruit", false, false),
        (Some(agents), "Agent", true, true),
        (predators, "Predator", true, false),
        (trees, "Tree", false, false),
    ] {
        let Some(objects) = objects else {
            continue;
        };
        let objects: Vec<_> = items(objects)?
            .into_iter()
            .filter(|o| !o.is(creature))
            .collect();
        if objects.is_empty() {
            continue;
        }
        let xs = objects
            .iter()
            .map(|o| f(o, "x"))
            .collect::<PyResult<Vec<_>>>()?;
        let ys = objects
            .iter()
            .map(|o| f(o, "y"))
            .collect::<PyResult<Vec<_>>>()?;
        let dx: Vec<_> = xs.iter().map(|v| v - x).collect();
        let dy: Vec<_> = ys.iter().map(|v| v - y).collect();
        let distances = binary(py, "hypot", dx.clone(), dy.clone())?;
        let angles: Vec<_> = binary(py, "arctan2", dy, dx)?
            .iter()
            .map(|a| wrap(a - dir))
            .collect();
        let mut selected: Vec<usize> = (0..objects.len())
            .filter(|&i| distances[i] <= hearing)
            .collect();
        let candidates: Vec<usize> = (0..objects.len())
            .filter(|&i| {
                distances[i] > hearing && distances[i] <= vision && angles[i].abs() <= cone / 2.0
            })
            .collect();
        if !candidates.is_empty() {
            if shape.is_none() {
                let mut vertices = vec![(x, y)];
                vertices.extend(&poly);
                shape = Some(
                    py.import("shapely.geometry")?
                        .getattr("Polygon")?
                        .call1((vertices,))?,
                );
            }
            let cx: Vec<_> = candidates.iter().map(|&i| xs[i]).collect();
            let cy: Vec<_> = candidates.iter().map(|&i| ys[i]).collect();
            let inside: Vec<bool> = py
                .import("shapely")?
                .getattr("contains_xy")?
                .call1((shape.as_ref().unwrap(), cx, cy))?
                .extract()?;
            selected.extend(
                candidates
                    .into_iter()
                    .zip(inside)
                    .filter_map(|(i, yes)| yes.then_some(i)),
            );
        }
        for i in selected {
            let obs = PyDict::new(py);
            obs.set_item("type", tag)?;
            obs.set_item("distance", distances[i])?;
            obs.set_item("angle", angles[i])?;
            if include_direction {
                let a = np2(py, "arctan2", y - ys[i], x - xs[i])?;
                obs.set_item("rel_dir", wrap(a - f(&objects[i], "direction")?))?;
            }
            if include_id {
                obs.set_item("id", objects[i].getattr("agent_id")?)?;
            }
            out.append(obs)?;
        }
    }
    let (c, s) = (np1(py, "cos", -dir)?, np1(py, "sin", -dir)?);
    for ((sx, sy), (ex, ey)) in hits {
        let (dxs, dys, dxe, dye) = (sx - x, sy - y, ex - x, ey - y);
        let obs = PyDict::new(py);
        obs.set_item("type", "Edge")?;
        obs.set_item(
            "coords",
            (
                (dxs * c - dys * s, dxs * s + dys * c),
                (dxe * c - dye * s, dxe * s + dye * c),
            ),
        )?;
        out.append(obs)?;
    }
    Ok(out)
}

pub fn blocked(point: Point, radius: f64, obstacles: &[(f64, f64, f64, f64)]) -> bool {
    obstacles.iter().any(|&(x, y, w, h)| {
        x - radius < point.0
            && point.0 < x + w + radius
            && y - radius < point.1
            && point.1 < y + h + radius
    })
}

pub fn rectangles(obstacles: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64, f64, f64)>> {
    items(obstacles)?
        .iter()
        .map(|o| Ok((f(o, "x")?, f(o, "y")?, f(o, "width")?, f(o, "height")?)))
        .collect()
}
