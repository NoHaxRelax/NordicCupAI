# Survival Simulator — evaluated submission

This branch contains only the Survival Simulator solution submitted by **Powered by Smørrebrød**. The other challenges are on the separate `jury/drone-task` and `jury/medical-appointment` branches.

Official evaluation score: **43,061.38441631117**, with no reported errors.

## Start here

1. Read [the HTTP entry point](nightsim/serve/fast1m_server.py) to see how an observation becomes a response.
2. Read [the observation adapter](nightsim/serve/night_policy.py) and [C++ survival policy](nightsim/_npolicy.hpp) for ordinary movement, feeding and reproduction. The submitted parameters are in [pred_best.json](nightsim/serve/pred_best.json).
3. Read [the repeated-action implementation](nightsim/serve/harvest.py) and [predator-contact predictor](nightsim/serve/contact_predictor.py) for the scoring mechanism described below.

The `nightsim` directory is the Python package containing our C++ implementation. Original source filenames and API route names are retained to keep the evaluated code unchanged.

## Decision process and scoring

For each request, the server passes the supplied agent observations to the C++ survival policy. This produces ordinary actions for movement, feeding, predator avoidance and reproduction. A second layer estimates whether a predator will reach an agent, using current and previous observations and actions. Wall observations help estimate the observer's movement between requests.

The second layer intentionally exploits a duplicate-action/negative-energy defect in the game engine:

1. Repeated stationary-turn actions for one agent drain its energy far below zero.
2. Another agent, immediately preceding it in the engine's agent list, is drained enough to die. Removing that agent while iterating the list can skip the following agent's energy-death check.
3. A predator eats the surviving negative-energy agent. The score calculation subtracts its negative energy, increasing the score. The predator also receives negative energy.

The submitted settings permit 1,000,000 repeated drain actions per attempt, at most 100 attempts per game, no cooldown, contact margin 1.0 and a minimum of six free agents. A response containing repeated actions can be about 109 MB. These settings are in `fast1m_server.py` and `harvest.py`.

The server does not recover the game seed or deliberately delay responses. The fixed seed 0 used by the observation adapter initializes its native policy container; received observations are passed to `policy_act_ext`. The included native engine also supports local simulation, but the serving path uses the observation adapter.

## Build and run

The deployed environment was Linux with Python **3.14.4**, NumPy **2.3.5**, FastAPI **0.121.2**, Uvicorn **0.38.0** and GCC **15.2.0**. A C++17 compiler and Python development headers are required.

From the repository root:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python nightsim/build.py
export PYTHONPATH="$PWD"
export FAST1M_LOG="$PWD/requests.jsonl"
cd nightsim/serve
../../.venv/bin/python -m uvicorn fast1m_server:app --host 0.0.0.0 --port 9064 --workers 1 --no-access-log
```

Send challenge observation JSON to `POST /predict`. The original submitted route, `/seed-live-mode144/predict`, is also supported. Both return an object containing an `actions` list. `GET /health` reports the current game and server counters.

Use **one worker**. Policy state is kept in memory and resets when simulation time decreases. Sequential games are supported; interleaved games are not isolated.

## Supporting files

| File | Purpose |
| --- | --- |
| `nightsim/_nengine.cpp` | C++ engine and Python extension interface |
| `nightsim/_npdp.hpp`, `nightsim/_nstuck.hpp` | Additional policy behaviors compiled with the survival policy; the supplied configuration selects behavior |
| `nightsim/__init__.py` | Python interface to the native extension |
| `nightsim/build.py` | Builds the extension from the included C++ source |
| `nightsim/serve/telemetry.py` | Bounded background logging of requests and timings; large outgoing action lists are not logged |
| `tests/test_contact_predictor.py` | Existing regression checks for predator-contact prediction |

Run the contact-prediction checks from the repository root with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Source correspondence

The production source and configuration match deployment commit `12fa6ff0fa4e73dd8a98d550529ecc778ba264de`. They were compared against the Hetzner deployment using SHA-256 when preparing the submission. This branch moves those files to the repository root, removes unrelated research and challenge files, and adds reviewer documentation. Only the test's import path was adjusted for its new location; production code is unchanged.
