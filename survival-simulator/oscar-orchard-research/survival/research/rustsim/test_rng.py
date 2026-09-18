"""Differential test: rustsim.PyRandom must reproduce random.Random draw for draw."""
import random, sys
import rustsim
ok = True
for seed in (0, 1, 7, 42, 2**31-1, 2**32+5, 123456789012345):
    py = random.Random(seed); rs = rustsim.PyRandom(seed)
    for i in range(20000):
        kind = i % 5
        if kind == 0: a, b = py.random(), rs.random()
        elif kind == 1: a, b = py.uniform(-3.5, 1600.0), rs.uniform(-3.5, 1600.0)
        elif kind == 2: a, b = py.getrandbits(11), rs.getrandbits(11)
        elif kind == 3: a, b = py.randint(0, 1599), rs.randint(0, 1599)
        else: n = 3 + (i % 4); a, b = py.choice(range(n)), rs.choice_index(n)
        if a != b: print(f"MISMATCH seed={seed} i={i} kind={kind} py={a!r} rs={b!r}"); ok = False; break
    # big getrandbits path (k > 32) and randint over wide range
    for i in range(2000):
        a, b = py.getrandbits(53), rs.getrandbits(53)
        if a != b: print(f"MISMATCH 53-bit seed={seed} i={i} py={a} rs={b}"); ok = False; break
        a, b = py.randint(0, 2**40), rs.randint(0, 2**40)
        if a != b: print(f"MISMATCH randint40 seed={seed} i={i} py={a} rs={b}"); ok = False; break
print("RNG PORT EXACT" if ok else "RNG PORT DIFFERS")
sys.exit(0 if ok else 1)
