use seed_search::{search, Signature};
use serde_json::{json, Value};
use std::{
    env, fs,
    io::{self, Write},
    path::PathBuf,
    process,
    time::Instant,
};

const HELP: &str = "seed-search --input FILE [--target N] [--checkpoint SECONDS]
  [--start N] [--count N] [--threads N] [--max-results N] [--output FILE]

Input: a survey probe result, or {\"samples\":[{\"x\":100,\"y\":200,\"biome\":\"forest\",\"radius\":7}]}.
--target selects a record in a multi-target survey; it is NOT a seed hint.
Defaults: start=0, count=10000000, checkpoint=latest, threads=available CPUs,
max-results=100000. Result JSON goes to stdout, or a new --output file.
All requested seeds are checked even if the result list is truncated.
This filters the native 1600x1200 ten-site land map; survivors need verification.";

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let (mut input, mut output) = (None, None);
    let (mut target, mut checkpoint) = (None, None);
    let (mut start, mut count) = (0u64, 10_000_000u64);
    let mut threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(256);
    let mut max_results = 100_000usize;
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!("{HELP}");
            return Ok(());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--target" => target = Some(value.parse::<u64>().map_err(|_| "invalid target")?),
            "--checkpoint" => {
                checkpoint = Some(value.parse::<f64>().map_err(|_| "invalid checkpoint")?)
            }
            "--start" => start = value.parse::<u64>().map_err(|_| "invalid start")?,
            "--count" => count = value.parse::<u64>().map_err(|_| "invalid count")?,
            "--threads" => threads = value.parse::<usize>().map_err(|_| "invalid threads")?,
            "--max-results" => {
                max_results = value.parse::<usize>().map_err(|_| "invalid max-results")?
            }
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    if checkpoint.is_some_and(|value| !value.is_finite() || value < 0.) {
        return Err("checkpoint must be a nonnegative finite number".into());
    }
    if output.as_ref().is_some_and(|path| path.exists()) {
        return Err("output already exists; choose a new file".into());
    }
    let input = input.ok_or("--input is required; use --help for usage")?;
    let content =
        fs::read(&input).map_err(|error| format!("cannot read {}: {error}", input.display()))?;
    let document: Value =
        serde_json::from_slice(&content).map_err(|error| format!("invalid JSON: {error}"))?;
    let signature = Signature::from_json(&document, target, checkpoint)?;
    let clock = Instant::now();
    let result = search(&signature, start, count, threads, max_results)?;
    let elapsed = clock.elapsed().as_secs_f64();
    let value = json!({
        "complete": true, "algorithm": "cpython-u32-biome-prefix-v1",
        "version": env!("CARGO_PKG_VERSION"), "input": input,
        "selected_target_record": signature.selected_target,
        "checkpoint_seconds": signature.checkpoint_seconds,
        "sample_count": signature.samples.len(), "ignored_river_samples": signature.ignored_river_samples,
        "seed_start": start, "seed_count": count, "processed_seeds": count,
        "threads": result.threads, "elapsed_seconds": elapsed,
        "seeds_per_second": count as f64 / elapsed,
        "survivor_count": result.total_survivors,
        "survivors_truncated": result.total_survivors > result.seeds.len() as u64,
        "survivors": result.seeds,
        "verification": "Prefix candidates only; validate using headings, rocks and native action replay."
    });
    let mut writer: Box<dyn Write> = match output {
        Some(path) => Box::new(
            fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)
                .map_err(|error| format!("cannot create {}: {error}", path.display()))?,
        ),
        None => Box::new(io::stdout().lock()),
    };
    serde_json::to_writer_pretty(&mut writer, &value).map_err(|error| error.to_string())?;
    writeln!(writer).map_err(|error| error.to_string())?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("seed-search: {error}");
        process::exit(1);
    }
}
