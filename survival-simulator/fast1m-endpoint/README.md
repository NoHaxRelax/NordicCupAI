# Survival Simulator — final submission

This is the source and configuration used by the evaluated endpoint. All included source files match the Hetzner deployment by SHA-256.

Official evaluation score: **43,061.38441631117**, with no reported errors.
The policy source is unchanged from deployment commit `12fa6ff0fa4e73dd8a98d550529ecc778ba264de`.

## How it works

`nightsim/serve/fast1m_server.py` receives game observations and returns agent actions. It runs the native survival policy, then applies the harvesting layer. When simulation time goes backwards, it resets state for a new game. Run with one worker; games must arrive sequentially.

`night_policy.py` converts observations into the native policy format. `pred_best.json` contains the exact policy parameters. `_npolicy.hpp`, `_npdp.hpp` and `_nstuck.hpp` contain the policy and helper behaviors; `_nengine.cpp` exposes the native engine and policy to Python. `__init__.py` provides the Python interface and `build.py` compiles the extension.

`harvest.py` and `contact_predictor.py` exploit the duplicate-action/negative-energy game defect. Repeated stationary turns drain an agent below zero. Removing the preceding agent during the engine's list iteration can skip the drained agent's energy-death check. When a predator eats that negative-energy agent, the score calculation subtracts negative energy, increasing the score. Contact prediction uses received observations and prior actions. The deployed configuration allows 1,000,000 drain actions per burst, up to 100 bursts per game, with cooldown 0, contact margin 1.0 and minimum free agents 6.

The serving code does not recover the game seed or deliberately delay responses. The legacy URL names do not describe the deployed strategy. The fixed seed 0 in NightPolicy initializes its native policy container; incoming observations are passed to policy_act_ext.

`telemetry.py` logs bounded request data and timing information through a background process. It does not log the large outgoing action payloads. `research/action_bug_debug/test_contact_predictor.py` contains the existing contact-prediction regression checks.

## Build and run

The deployment used Python 3.14.4, NumPy 2.3.5, FastAPI 0.121.2, Uvicorn 0.38.0 and GCC 15.2.0 on Linux. Install a C++17 compiler and Python development headers, then run from this directory:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install numpy==2.3.5 fastapi==0.121.2 uvicorn==0.38.0
.venv/bin/python nightsim/build.py
export PYTHONPATH="$PWD"
export FAST1M_LOG="$PWD/fast1m.jsonl"
cd nightsim/serve
../../.venv/bin/python -m uvicorn fast1m_server:app --host 0.0.0.0 --port 9064 --workers 1 --no-access-log
```

Send the challenge observation JSON to `POST /predict` or `/seed-live-mode144/predict`. The response is an object containing an `actions` list. `GET /health` reports current state.
