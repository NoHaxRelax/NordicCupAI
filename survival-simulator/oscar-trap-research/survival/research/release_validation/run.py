from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research'),str(ROOT/'research/simple_chase')]
from simple_chase import run_streaming_v2 as harness
harness.OUT=ROOT/'results/release_validation'
if __name__=='__main__':harness.main()
