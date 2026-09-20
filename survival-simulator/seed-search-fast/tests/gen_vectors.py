"""Emit ground-truth terrain prefixes using CPython's own random.Random.

This is the authority the C++ kernels are checked against: it mirrors the exact call
sequence in src/elements/biome.py Map_generator.generate (num_biomes=10, 1600x1200),
as instantiated at src/elements/environment.py:38.

    python gen_vectors.py > vectors.txt

Output lines: SEED x0 y0 x1 y1 ... x9 y9 t0 t1 ... t9
"""
import random
import sys

WIDTH, HEIGHT, NUM_BIOMES = 1600, 1200, 10

# Seeds worth pinning: the domain edges, small integers, the seed V4 recovered live,
# the V4 regression fixture seed, and a spread of large values.
SEEDS = (
    list(range(0, 64))
    + [2**32 - 1, 2**32 - 2, 2**31, 2**31 - 1, 65535, 65536]
    + [614466944, 1854492595, 78431, 20260919, 1001]
    + [i * 61_000_003 % (2**32) for i in range(1, 200)]
    # Multi-word keys: random.Random(int) splits |n| into little-endian 32-bit words, so
    # these exercise the K-word init_by_array path the scanner needs for seeds >= 2^32.
    + [7 + 2**32, 614466944 + 2**32 * 3, 2**64 + 12345, 2**63 + 2**32 + 99, 2**96 + 5]
)


def prefix(seed: int):
    rng = random.Random(seed)
    # biome.py: points = [(rng.randint(0, w-1), rng.randint(0, h-1)) for _ in range(n)]
    points = [(rng.randint(0, WIDTH - 1), rng.randint(0, HEIGHT - 1)) for _ in range(NUM_BIOMES)]
    # biome.py: biomes = [rng.choice(self.biome_types)() for _ in range(n)]
    # choice() draws _randbelow(4); a 4-element list of indices consumes the identical stream.
    types = [rng.choice([0, 1, 2, 3]) for _ in range(NUM_BIOMES)]
    return points, types


def main() -> int:
    seen = set()
    for seed in SEEDS:
        if seed in seen:
            continue
        seen.add(seed)
        points, types = prefix(seed)
        # Emit the key as CPython would split it: little-endian 32-bit words, comma-joined.
        # The C++ test parses either a bare integer (one word) or "w0,w1,...".
        words = []
        n = abs(seed) or 0
        while True:
            words.append(str(n & 0xFFFFFFFF))
            n >>= 32
            if n == 0:
                break
        flat = [",".join(words)]
        for x, y in points:
            flat += [str(x), str(y)]
        flat += [str(t) for t in types]
        print(" ".join(flat))
    print(f"# {len(seen)} vectors from CPython {sys.version.split()[0]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
