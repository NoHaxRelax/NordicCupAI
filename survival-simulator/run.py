"""Supported survival commands. Use '<command> --help' for its options."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "trapping": "scripts/trapping_game.py",
    "view": "scripts/entrapment_viewer.py",
    "orchard": "scripts/run_orchard.py",
    "benchmark": "scripts/entrapment_benchmark.py",
    "benchmark-view": "scripts/entrapment_benchmark_viewer.py",
    "benchmark-merge": "scripts/entrapment_benchmark_merge.py",
    "serve": "agent_server.py",
    "tune": "scripts/optimize_policy.py",
    "research": "scripts/research_loop.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "trapping: record a game with predators; view: replay it in a browser; "
        "orchard: no-predator survival (add --render for a window)."))
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args(sys.argv[1:2])
    target = ROOT / COMMANDS[args.command]
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
