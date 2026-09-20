"""Create a small source bundle for a CPU worker, excluding runs and credentials."""
from pathlib import Path
import argparse
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = [ROOT / "scripts/predator_stuck_scan.py", ROOT / "src/core.py",
             ROOT / "src/utils/simulation.py", ROOT / "src/utils/sensing.py",
             *sorted((ROOT / "src/elements").glob("*.py"))]
    for relative in ("src/__init__.py", "src/utils/__init__.py"):
        if (ROOT / relative).exists():
            paths.append(ROOT / relative)
    paths += [p for p in Path(__file__).parent.iterdir() if p.suffix in (".py", ".cpp", ".txt", ".md", ".sh")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as tar:
        for path in paths:
            tar.add(path, arcname=path.relative_to(ROOT), recursive=False)
    print(f"Packed {len(paths)} source files, {args.output.stat().st_size} bytes: {args.output}")


if __name__ == "__main__":
    main()
