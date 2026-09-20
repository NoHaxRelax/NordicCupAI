"""Measure MT state-equation rank from ideal, consecutive Python random() outputs.

This is an information-budget experiment, not a public-geometry seed solver.
It follows CPython 3.12's MT recurrence and checks symbolic outputs against the
installed random.Random. Unknown state bits are variables before the first twist.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import time


def twist(state):
    for i in range(624):
        mixed = state[(i + 1) % 624][:31] + state[i][31:]
        distant = state[(i + 397) % 624]
        state[i] = [distant[bit] ^ (mixed[bit + 1] if bit < 31 else 0)
                    ^ (mixed[0] if (0x9908B0DF >> bit) & 1 else 0)
                    for bit in range(32)]


def temper(word):
    word = [word[i] ^ (word[i + 11] if i + 11 < 32 else 0) for i in range(32)]
    word = [word[i] ^ (word[i - 7] if i >= 7 and (0x9D2C5680 >> i) & 1 else 0)
            for i in range(32)]
    word = [word[i] ^ (word[i - 15] if i >= 15 and (0xEFC60000 >> i) & 1 else 0)
            for i in range(32)]
    return [word[i] ^ (word[i + 18] if i + 18 < 32 else 0) for i in range(32)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-floats", type=int, default=1500)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output filename")
    if args.max_floats < 1:
        parser.error("--max-floats must be positive")

    control = random.Random(20260919)
    assignment = sum(word << (i * 32) for i, word in enumerate(control.getstate()[1][:-1]))
    state = [[1 << (i * 32 + bit) for bit in range(32)] for i in range(624)]
    basis = [0] * (624 * 32)
    rank = 0
    rows = []
    checkpoint = {80, 160, 320, 377, 400, 500, 600, 623, 624, 650, 700, 1000, 1500}
    started = time.perf_counter()
    for sample in range(1, args.max_floats + 1):
        for half, discarded in enumerate((5, 6)):
            output_index = (sample - 1) * 2 + half
            index = output_index % 624
            if index == 0:
                twist(state)
            word = temper(state[index])
            expected = control.getrandbits(32)
            # Check every symbolic word against CPython, including twist boundaries.
            actual = sum(((expression & assignment).bit_count() & 1) << bit
                         for bit, expression in enumerate(word))
            if actual != expected:
                raise AssertionError(f"MT mismatch at word {output_index}")
            for expression in reversed(word[discarded:]):
                while expression:
                    pivot = expression.bit_length() - 1
                    if basis[pivot]:
                        expression ^= basis[pivot]
                    else:
                        basis[pivot] = expression
                        rank += 1
                        break
        rows.append(dict(floats=sample, exposed_bits=sample * 53, rank=rank))
        if sample in checkpoint or rank == 19937:
            print(f"{sample} floats: rank {rank}/19937", flush=True)
        if rank == 19937:
            break
    result = dict(
        complete=True, python=platform.python_version(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter() - started,
        information_count_lower_bound=377,
        first_full_rank_float_count=rows[-1]["floats"] if rank == 19937 else None,
        cpython_words_verified=len(rows) * 2, ranks=rows,
        assumptions=[
            "Exact original random() values, exposing alternating 27 and 26 output bits.",
            "Consecutive outputs starting immediately after a known twist boundary.",
            "Unrestricted valid MT state; no 32-bit integer seed constraint.",
            "No geometric rounding, unknown ordering, missing draws or observation selection.",
            "Rank is measured; a state/seed inversion solver is not implemented here.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
