//! Native 1600x1200 biome-prefix filter, compatible with CPython integer seeding.
//! MT constants/recurrence: CPython Modules/_randommodule.c (v3.12.12).
//! Rejection sampling: CPython Lib/random.py, _randbelow_with_getrandbits.

use serde_json::Value;
use std::thread;

const N: usize = 624;
pub const SEED_SPACE: u64 = 1u64 << 32;

const fn initial_state() -> [u32; N] {
    let mut state = [0u32; N];
    state[0] = 19_650_218;
    let mut i = 1;
    while i < N {
        state[i] = (state[i - 1] ^ (state[i - 1] >> 30))
            .wrapping_mul(1_812_433_253)
            .wrapping_add(i as u32);
        i += 1;
    }
    state
}

const INITIAL_STATE: [u32; N] = initial_state();

pub struct PythonRandom {
    state: [u32; N],
    index: usize,
}

impl PythonRandom {
    pub fn new(seed: u32) -> Self {
        // Python's nonnegative <=32-bit integer seed is a one-word init_key.
        // The first init_genrand pass is constant across all such seeds.
        let mut state = INITIAL_STATE;
        let mut i = 1;
        for _ in 0..N {
            state[i] = (state[i] ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(seed);
            i += 1;
            if i == N {
                state[0] = state[N - 1];
                i = 1;
            }
        }
        for _ in 0..N - 1 {
            state[i] = (state[i]
                ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i == N {
                state[0] = state[N - 1];
                i = 1;
            }
        }
        state[0] = 0x8000_0000;
        Self { state, index: 0 }
    }

    pub fn next_u32(&mut self) -> u32 {
        // Advance the in-place twist lazily. Prefix generation usually needs
        // far fewer than 624 words. This also works across complete cycles.
        let i = self.index;
        let next = if i + 1 == N { 0 } else { i + 1 };
        let distant = if i < 227 { i + 397 } else { i - 227 };
        let mixed = (self.state[i] & 0x8000_0000) | (self.state[next] & 0x7fff_ffff);
        let mut word = self.state[distant] ^ (mixed >> 1);
        if mixed & 1 != 0 {
            word ^= 0x9908_b0df;
        }
        self.state[i] = word;
        self.index = next;
        word ^= word >> 11;
        word ^= (word << 7) & 0x9d2c_5680;
        word ^= (word << 15) & 0xefc6_0000;
        word ^ (word >> 18)
    }

    pub fn randbelow(&mut self, limit: u32) -> u32 {
        assert!(limit > 0);
        let bits = 32 - limit.leading_zeros(); // n.bit_length(), including powers of two.
        loop {
            let value = self.next_u32() >> (32 - bits);
            if value < limit {
                return value;
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Site {
    pub x: f64,
    pub y: f64,
    pub biome: u8,
}

pub fn candidate_prefix(seed: u32) -> [Site; 10] {
    let mut rng = PythonRandom::new(seed);
    let mut sites = [Site {
        x: 0.,
        y: 0.,
        biome: 0,
    }; 10];
    for site in &mut sites {
        site.x = rng.randbelow(1600) as f64;
        site.y = rng.randbelow(1200) as f64;
    }
    // All coordinates precede ALL type choices in Map_generator.generate().
    for site in &mut sites {
        site.biome = rng.randbelow(4) as u8;
    }
    sites
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub biome: u8,
}

pub struct Signature {
    pub samples: Vec<Sample>,
    required_types: u8,
    pub ignored_river_samples: usize,
    pub selected_target: Option<u64>,
    pub checkpoint_seconds: Option<f64>,
}

impl Signature {
    pub fn from_json(
        root: &Value,
        target: Option<u64>,
        checkpoint: Option<f64>,
    ) -> Result<Self, String> {
        let (raw, selected_target, checkpoint_seconds) = if let Some(samples) = root.get("samples")
        {
            if target.is_some() || checkpoint.is_some() {
                return Err("--target and --checkpoint apply only to survey result files".into());
            }
            (samples, None, None)
        } else {
            let targets = root["targets"]
                .as_array()
                .ok_or("expected samples or survey targets")?;
            let row = match target {
                Some(seed) => targets
                    .iter()
                    .find(|row| row["target_seed"].as_u64() == Some(seed))
                    .ok_or("selected target record does not exist")?,
                None if targets.len() == 1 => &targets[0],
                _ => {
                    return Err("survey has multiple targets; select a record with --target".into())
                }
            };
            let checkpoints = row["checkpoints"]
                .as_array()
                .ok_or("missing survey checkpoints")?;
            let frame = match checkpoint {
                Some(seconds) => checkpoints
                    .iter()
                    .find(|frame| {
                        frame["sim_time"]
                            .as_f64()
                            .is_some_and(|time| (time - seconds).abs() <= 1e-6)
                    })
                    .ok_or("selected checkpoint does not exist")?,
                None => checkpoints.last().ok_or("survey has no checkpoints")?,
            };
            (
                &frame["shared_samples"],
                row["target_seed"].as_u64(),
                frame["sim_time"].as_f64(),
            )
        };
        let mut samples = Vec::new();
        let mut required_types = 0;
        let mut ignored_river_samples = 0;
        for (index, row) in raw
            .as_array()
            .ok_or("samples must be an array")?
            .iter()
            .enumerate()
        {
            let biome = match row["biome"].as_str() {
                Some("forest") => 0,
                Some("swamp") => 1,
                Some("desert") => 2,
                Some("grassland") => 3,
                Some("river") => {
                    ignored_river_samples += 1;
                    continue;
                }
                _ => return Err(format!("sample {index}: unknown biome")),
            };
            let number = |key: &str| {
                row[key]
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| format!("sample {index}: {key} must be a finite number"))
            };
            let (x, y, radius) = (number("x")?, number("y")?, number("radius")?);
            let max_distance = (x.abs() + 1600.).hypot(y.abs() + 1200.) + 2. * radius;
            if radius < 0. || !(max_distance * max_distance).is_finite() {
                return Err(format!(
                    "sample {index}: invalid radius or coordinates too large"
                ));
            }
            required_types |= 1 << biome;
            samples.push(Sample {
                x,
                y,
                radius,
                biome,
            });
        }
        if samples.is_empty() {
            return Err(
                "no non-river samples; there is no evidence to narrow the seed range".into(),
            );
        }
        Ok(Self {
            samples,
            required_types,
            ignored_river_samples,
            selected_target,
            checkpoint_seconds,
        })
    }

    pub fn matches(&self, sites: &[Site; 10]) -> bool {
        let available = sites
            .iter()
            .fold(0u8, |mask, site| mask | (1 << site.biome));
        if available & self.required_types != self.required_types {
            return false;
        }
        for sample in &self.samples {
            let (mut closest, mut desired) = (f64::INFINITY, f64::INFINITY);
            for site in sites {
                let dx = sample.x - site.x;
                let dy = sample.y - site.y;
                let squared = dx * dx + dy * dy;
                closest = closest.min(squared);
                if site.biome == sample.biome {
                    desired = desired.min(squared);
                }
            }
            let bound = closest.sqrt() + 2. * sample.radius;
            if desired > bound * bound + 1e-9 {
                return false;
            }
        }
        true
    }
}

#[derive(Debug, PartialEq)]
pub struct SearchResult {
    pub seeds: Vec<u32>,
    pub total_survivors: u64,
    pub threads: usize,
}

pub fn search(
    signature: &Signature,
    start: u64,
    count: u64,
    threads: usize,
    max_results: usize,
) -> Result<SearchResult, String> {
    if start >= SEED_SPACE || count == 0 || count > SEED_SPACE - start {
        return Err("range must be nonempty and stay within unsigned 32-bit seeds".into());
    }
    if !(1..=256).contains(&threads) || max_results == 0 {
        return Err("threads must be 1..256 and max-results must be positive".into());
    }
    let threads = threads.min(count.min(256) as usize);
    let batches = thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..threads {
            let begin = start + count * worker as u64 / threads as u64;
            let end = start + count * (worker + 1) as u64 / threads as u64;
            handles.push(scope.spawn(move || {
                let mut seeds = Vec::new();
                let mut total = 0u64;
                for seed in begin..end {
                    if signature.matches(&candidate_prefix(seed as u32)) {
                        total += 1;
                        if seeds.len() < max_results {
                            seeds.push(seed as u32);
                        }
                    }
                }
                (seeds, total)
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| "search worker panicked".to_string())
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    let mut result = SearchResult {
        seeds: Vec::new(),
        total_survivors: 0,
        threads,
    };
    for (seeds, total) in batches {
        result.seeds.extend(seeds);
        result.total_survivors += total;
    }
    result.seeds.sort_unstable();
    result.seeds.truncate(max_results);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixtures() -> Value {
        serde_json::from_str(include_str!("../tests/python_vectors.json")).unwrap()
    }

    #[test]
    fn raw_stream_matches_python_across_twist_boundaries() {
        for fixture in fixtures()["rng_vectors"].as_array().unwrap() {
            let mut rng = PythonRandom::new(fixture["seed"].as_u64().unwrap() as u32);
            let expected = fixture["outputs"].as_array().unwrap();
            let mut check = 0;
            for index in 0..2000 {
                let actual = rng.next_u32();
                if check < expected.len() && expected[check][0].as_u64().unwrap() == index {
                    assert_eq!(
                        actual as u64,
                        expected[check][1].as_u64().unwrap(),
                        "seed {}, word {index}",
                        fixture["seed"]
                    );
                    check += 1;
                }
            }
            assert_eq!(check, expected.len());
        }
    }

    #[test]
    fn generated_sites_and_types_match_python() {
        for fixture in fixtures()["prefix_vectors"].as_array().unwrap() {
            let actual = candidate_prefix(fixture["seed"].as_u64().unwrap() as u32);
            for (index, site) in actual.iter().enumerate() {
                assert_eq!(site.x, fixture["sites"][index][0].as_f64().unwrap());
                assert_eq!(site.y, fixture["sites"][index][1].as_f64().unwrap());
                assert_eq!(site.biome as u64, fixture["types"][index].as_u64().unwrap());
            }
        }
    }

    #[test]
    fn scans_match_python_with_parallelism_and_truncated_output() {
        for fixture in fixtures()["search_vectors"].as_array().unwrap() {
            let signature = Signature::from_json(fixture, None, None).unwrap();
            let expected: Vec<u32> = fixture["expected"]
                .as_array()
                .unwrap()
                .iter()
                .map(|seed| seed.as_u64().unwrap() as u32)
                .collect();
            for threads in [1, 3, 8] {
                for limit in [2, 1000] {
                    let result = search(
                        &signature,
                        fixture["start"].as_u64().unwrap(),
                        fixture["count"].as_u64().unwrap(),
                        threads,
                        limit,
                    )
                    .unwrap();
                    assert_eq!(result.total_survivors as usize, expected.len());
                    assert_eq!(
                        result.seeds,
                        expected.iter().copied().take(limit).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_or_empty_evidence_is_rejected() {
        for value in [
            json!({"samples": []}),
            json!({"samples": [{"biome":"river"}]}),
            json!({"samples": [{"x":0,"y":0,"biome":"typo","radius":1}]}),
            json!({"samples": [{"x":0,"y":0,"biome":"forest","radius":-1}]}),
            json!({"samples": [{"x":0,"y":0,"biome":"forest"}]}),
        ] {
            assert!(Signature::from_json(&value, None, None).is_err());
        }
    }

    #[test]
    fn invalid_ranges_are_rejected() {
        let signature = Signature::from_json(&fixtures()["search_vectors"][0], None, None).unwrap();
        for (start, count, threads, limit) in [
            (0, 0, 1, 10),
            (SEED_SPACE, 1, 1, 10),
            (SEED_SPACE - 1, 2, 1, 10),
            (0, 1, 0, 10),
            (0, 1, 257, 10),
            (0, 1, 1, 0),
        ] {
            assert!(search(&signature, start, count, threads, limit).is_err());
        }
    }

    #[test]
    fn uncertainty_bound_and_missing_biomes_are_respected() {
        let mut sites = [Site {
            x: 10.,
            y: 0.,
            biome: 0,
        }; 10];
        sites[9] = Site {
            x: 20.,
            y: 0.,
            biome: 1,
        };
        for (radius, expected) in [(5., true), (4.9, false)] {
            let signature = Signature::from_json(
                &json!({"samples": [
                    {"x":0,"y":0,"biome":"swamp","radius":radius}, {"biome":"river"}
                ]}),
                None,
                None,
            )
            .unwrap();
            assert_eq!(signature.ignored_river_samples, 1);
            assert_eq!(signature.matches(&sites), expected);
        }
        sites[9].biome = 0;
        let signature = Signature::from_json(
            &json!({"samples": [
                {"x":0,"y":0,"biome":"swamp","radius":10000}
            ]}),
            None,
            None,
        )
        .unwrap();
        assert!(!signature.matches(&sites));
    }

    #[test]
    fn survey_record_and_checkpoint_selection() {
        let samples = json!([{"x":100,"y":200,"biome":"forest","radius":7}]);
        let row = json!({"target_seed":3,"checkpoints":[
            {"sim_time":10,"shared_samples":samples},
            {"sim_time":30,"shared_samples":samples}
        ]});
        let document = json!({"targets":[row, {"target_seed":11,"checkpoints":[]}]});
        assert!(Signature::from_json(&document, None, None).is_err());
        assert!(Signature::from_json(&document, Some(4), None).is_err());
        assert!(Signature::from_json(&document, Some(11), None).is_err());
        assert!(Signature::from_json(&document, Some(3), Some(20.)).is_err());
        let latest = Signature::from_json(&document, Some(3), None).unwrap();
        assert_eq!(latest.checkpoint_seconds, Some(30.));
        let earlier = Signature::from_json(&document, Some(3), Some(10.)).unwrap();
        assert_eq!(earlier.checkpoint_seconds, Some(10.));
    }
}
