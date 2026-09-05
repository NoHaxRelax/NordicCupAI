"""Measure the simulator throughput that decides whether RL is viable."""

import random
import time

from racecar_env import ACTIONS, RaceCarEnv


def random_policy(_obs):
    return random.choice(ACTIONS)


def straight_policy(_obs):
    return "ACCELERATE"


def main() -> None:
    env = RaceCarEnv(seed=12345)

    print("=== termination sanity ===")
    for name, policy in (("random", random_policy), ("always-accelerate", straight_policy)):
        print(f"  {name:18s} {env.run_episode(policy)}")

    print("\n=== throughput ===")
    env.reset(1)
    n, t0 = 5000, time.perf_counter()
    steps = 0
    for _ in range(n):
        if env.step(random.choice(ACTIONS)).terminated:
            env.reset(random.randrange(10**6))
        steps += 1
    rate = steps / (time.perf_counter() - t0)
    print(f"  headless          {rate:>10,.0f} ticks/sec/core")
    print(f"  organisers' loop  {60:>10,} ticks/sec (clock.tick(60))")
    print(f"  speedup           {rate/60:>10,.0f}x")

    print("\n=== RL budget (10M env steps) ===")
    for cores in (4, 16, 32, 64):
        hours = 10e6 / (rate * cores) / 3600
        print(f"  {cores:>3d} cores      {hours:>8.2f} h")
    print(f"  realtime loop   {10e6/60/3600:>8.0f} h   <- the whole competition, twice over")


if __name__ == "__main__":
    main()
