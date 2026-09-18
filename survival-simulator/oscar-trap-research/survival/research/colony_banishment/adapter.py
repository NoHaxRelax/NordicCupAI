"""Factories for the existing debugger's --policy option. No new renderer."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from policy import ColonyPolicy


def make_colony(seed=0):
    return ColonyPolicy(banish=False)


def make_banishment(seed=0):
    return ColonyPolicy(banish=True)
