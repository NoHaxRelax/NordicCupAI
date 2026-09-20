"""Regression checks for unordered public observations, not executed actions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.seed_shadow.replay import canonical,close
fruit={'type':'Fruit','angle':0.,'distance':85.2}
agent={'type':'Agent','id':32,'angle':2.35,'distance':36.9}
a=canonical([fruit,agent]);b=canonical([{**fruit,'angle':1.7e-15},agent])
assert close(a,b)
assert not close(a,b+[fruit])
assert not close(a,[{**fruit,'distance':86.},agent])
assert not close(a,[fruit,{**agent,'id':33}])
assert not close([{'coords':[[0.,1.],[2.,3.]],'type':'Edge'}],[{'coords':[[2.,3.],[0.,1.]],'type':'Edge'}])
# Ambiguous tolerant matches need an augmenting path, not greedy pairing.
assert close([{'x':1e-8},{'x':0.}],[{'x':0.},{'x':2e-8}])
assert not close([{'x':0.},{'x':0.}],[{'x':0.},{'x':1.}])
print('PASS: order tolerance, multiplicity, identities, coordinates, and matching ambiguity')
