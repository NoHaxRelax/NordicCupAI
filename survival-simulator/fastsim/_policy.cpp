// fastsim._policy - the fastsim engine PLUS the native orchard + evasion policy,
// linked into one extension so a decision is a direct C++ call.
//
// Why a second extension module instead of adding the hooks to _engine.cpp:
// fastsim/_engine.cp312-win_amd64.pyd is a built artefact whose provenance is pinned
// by fastsim/build-info.json (source_sha256 of _engine.cpp, binary_sha256 of the .pyd).
// Editing _engine.cpp in place would invalidate that record and the benchmark numbers
// measured against it. So _engine.cpp is left byte-for-byte alone and #include'd here
// verbatim: this translation unit gets its own private copy of the engine, and is
// compiled to its own _policy<EXT_SUFFIX> with its own build-info-policy.json.
// `import fastsim` keeps using _engine exactly as before.
//
// The cost of the arrangement is that the two modules do not share engine state or
// the numpy ufunc loop pointers - fastsim/fastpolicy.py calls _policy.set_numpy_loops
// on import, the same way fastsim/__init__.py does for _engine.
//
// THIS file sees the engine. The POLICY does not: it is compiled separately
// (_orchard_policy.cpp) against policy_abi.hpp alone, and is reached only through the
// polabi::IPolicy vtable below. See policy_abi.hpp and fastsim/check_boundary.py.
//
// The Python bindings follow survival-simulator/oscar-fastsim 51ca0680's
// fastsim/_engine.cpp (policy_init / policy_act / policy_minds / policy_groups /
// run_policy), reworked to go through the narrow interface and to keep the per-tick
// path free of Python objects.

// _engine.cpp defines PyInit__engine; this module's init is PyInit__policy, so rename
// the included one out of the way rather than export a second, wrong-named init.
#define PyInit__engine PyInit__engine_not_used_here
#include "_engine.cpp"
#undef PyInit__engine

#include "policy_iface.hpp"

#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

// The observation list is handed to the policy without a copy. That is only sound if
// the engine's Obs and the boundary's polabi::Obs have the same layout, so assert it
// field by field - if _engine.cpp ever changes Obs, this build fails instead of
// silently feeding the policy garbage.
static_assert(sizeof(Obs) == sizeof(polabi::Obs), "Obs size");
static_assert(offsetof(Obs, type) == offsetof(polabi::Obs, type), "Obs.type");
static_assert(offsetof(Obs, distance) == offsetof(polabi::Obs, distance), "Obs.distance");
static_assert(offsetof(Obs, angle) == offsetof(polabi::Obs, angle), "Obs.angle");
static_assert(offsetof(Obs, rel_dir) == offsetof(polabi::Obs, rel_dir), "Obs.rel_dir");
static_assert(offsetof(Obs, has_rel_dir) == offsetof(polabi::Obs, has_rel_dir), "Obs.has_rel_dir");
static_assert(offsetof(Obs, id) == offsetof(polabi::Obs, id), "Obs.id");
static_assert(offsetof(Obs, has_id) == offsetof(polabi::Obs, has_id), "Obs.has_id");
static_assert(offsetof(Obs, c) == offsetof(polabi::Obs, c), "Obs.coords");
static_assert((int)RIVER == (int)polabi::RIVER && (int)FOREST == (int)polabi::FOREST &&
                  (int)SWAMP == (int)polabi::SWAMP && (int)DESERT == (int)polabi::DESERT &&
                  (int)GRASSLAND == (int)polabi::GRASSLAND,
              "biome indices");

// Same layout prefix as EngineObject (PyObject_HEAD then Engine*), so the engine's own
// binding functions can be reused unchanged by casting - the standard C-extension
// subclassing trick, and the reason the whole Engine surface needs no duplication.
struct PolicyEngineObject {
    PyObject_HEAD
    Engine* eng;
    polabi::IPolicy* pol;
    std::vector<polabi::AState>* buf;  // reused across ticks; never reallocated in steady state
};
static_assert(offsetof(PolicyEngineObject, eng) == offsetof(EngineObject, eng),
              "PolicyEngineObject must share EngineObject's layout prefix");

// The per-tick input. Only public observation fields are read out of the engine; the
// observation vector itself is passed by pointer, so nothing is serialized or boxed.
void fill_states(Engine* e, std::vector<polabi::AState>& out) {
    static const std::vector<Obs> empty;
    out.clear();
    for (const Creature& a : e->agents) {
        const std::vector<Obs>* obs = (a.id >= 0 && (size_t)a.id < e->agent_observations.size())
                                          ? &e->agent_observations[(size_t)a.id] : &empty;
        polabi::AState s;
        s.aid = a.id;
        s.obs = reinterpret_cast<const std::vector<polabi::Obs>*>(obs);  // layout asserted above
        s.energy = a.energy; s.biome = e->biome_at(a.x, a.y); s.age = a.age; s.speed = a.speed;
        s.sprint = a.sprint_speed; s.hear = a.hearing_radius; s.cone = a.cone_angle; s.vr = a.vision_radius;
        s.max_energy = a.max_energy;
        out.push_back(s);
    }
}

PyObject* PE_policy_init(PolicyEngineObject* self, PyObject* args) {
    PyObject *key_obj, *cfg = nullptr;
    int preserve_memory = 0;
    if (!PyArg_ParseTuple(args, "O|Op", &key_obj, &cfg, &preserve_memory)) return nullptr;
    if (preserve_memory && !self->pol) { PyErr_SetString(PyExc_RuntimeError, "policy_init first"); return nullptr; }
    PyObject* seq = PySequence_Fast(key_obj, "seed_key must be a sequence");
    if (!seq) return nullptr;
    std::vector<uint32_t> key;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); i++)
        key.push_back((uint32_t)PyLong_AsUnsignedLong(PySequence_Fast_GET_ITEM(seq, i)));
    Py_DECREF(seq);
    if (PyErr_Occurred()) return nullptr;

    // Flatten the config dict here, once, so the policy never touches a Python object.
    std::vector<std::string> names;
    std::vector<polabi::CfgItem> items;
    std::string feed;
    if (cfg && cfg != Py_None) {
        if (!PyDict_Check(cfg)) { PyErr_SetString(PyExc_TypeError, "config must be a dict"); return nullptr; }
        PyObject *k, *v; Py_ssize_t pos = 0;
        while (PyDict_Next(cfg, &pos, &k, &v)) {
            const char* ks = PyUnicode_AsUTF8(k);
            if (!ks) return nullptr;
            if (std::strcmp(ks, "feed_mode") == 0) {
                const char* sv = PyUnicode_AsUTF8(v);
                if (!sv) return nullptr;
                feed = sv;
                continue;
            }
            double x;
            if (PyBool_Check(v)) x = (v == Py_True) ? 1. : 0.;
            else {
                x = PyFloat_AsDouble(v);
                if (x == -1.0 && PyErr_Occurred()) { PyErr_Clear(); continue; }  // e.g. _debug_merge
            }
            names.push_back(ks);
            items.push_back(polabi::CfgItem{nullptr, x});
        }
        for (size_t i = 0; i < names.size(); i++) items[i].key = names[i].c_str();
    }
    polabi::Cfg c{items.data(), items.size(), feed.empty() ? nullptr : feed.c_str()};
    const char* err = nullptr;
    polabi::IPolicy* p = polabi::make_policy(key.data(), key.size(), c, &err);
    if (!p) { PyErr_SetString(PyExc_ValueError, err ? err : "policy configuration rejected"); return nullptr; }
    if (preserve_memory) {
        self->pol->copy_parameters(*p);
        polabi::destroy_policy(p);
    } else {
        polabi::destroy_policy(self->pol);
        self->pol = p;
    }
    Py_RETURN_NONE;
}

PyObject* acts_to_py(const std::vector<polabi::Act>& acts) {
    PyObject* list = PyList_New((Py_ssize_t)acts.size());
    for (size_t i = 0; i < acts.size(); i++) {
        const polabi::Act& a = acts[i];
        PyList_SET_ITEM(list, (Py_ssize_t)i,
                        Py_BuildValue("(LdddN)", (long long)a.aid, a.dist, a.direction, a.turn,
                                      PyBool_FromLong(a.spawn)));
    }
    return list;
}

// Decisions of the native policy for the current state (does not step). This is the
// lockstep-verification entry point, not the fast path: it builds Python tuples so the
// verifier can compare them. run_policy is the fast path and builds none.
PyObject* PE_policy_act(PolicyEngineObject* self, PyObject*) {
    if (!self->pol) { PyErr_SetString(PyExc_RuntimeError, "policy_init first"); return nullptr; }
    Engine* e = self->eng;
    fill_states(e, *self->buf);
    return acts_to_py(self->pol->call(self->buf->data(), self->buf->size(), e->time));
}

PyObject* PE_policy_minds(PolicyEngineObject* self, PyObject*) {
    if (!self->pol) { PyErr_SetString(PyExc_RuntimeError, "policy_init first"); return nullptr; }
    std::vector<polabi::MindRow> rows(self->pol->dump_minds(nullptr, 0));
    if (!rows.empty()) self->pol->dump_minds(rows.data(), rows.size());
    PyObject* list = PyList_New((Py_ssize_t)rows.size());
    for (size_t i = 0; i < rows.size(); i++) {
        const polabi::MindRow& r = rows[i];
        PyObject* t = Py_BuildValue(
            "(LLdddNNONNLddLLd)", (long long)r.aid, (long long)r.group, r.px, r.py, r.theta,
            r.has_post ? PyLong_FromLongLong(r.post) : (Py_INCREF(Py_None), Py_None),
            r.has_fruit ? PyLong_FromLongLong(r.fruit) : (Py_INCREF(Py_None), Py_None),
            r.old ? Py_True : Py_False,
            r.has_explore ? Py_BuildValue("(dd)", r.ex, r.ey) : (Py_INCREF(Py_None), Py_None),
            r.has_watch ? Py_BuildValue("(dd)", r.wx, r.wy) : (Py_INCREF(Py_None), Py_None),
            (long long)r.n_edges, r.best_d, r.energy_prev, (long long)r.n_prev_marks,
            (long long)r.n_hear_hist, r.prev_theta);
        PyList_SET_ITEM(list, (Py_ssize_t)i, t);
    }
    return list;
}

PyObject* PE_policy_groups(PolicyEngineObject* self, PyObject*) {
    if (!self->pol) { PyErr_SetString(PyExc_RuntimeError, "policy_init first"); return nullptr; }
    std::vector<polabi::GroupRow> rows(self->pol->dump_groups(nullptr, 0));
    if (!rows.empty()) self->pol->dump_groups(rows.data(), rows.size());
    PyObject* list = PyList_New((Py_ssize_t)rows.size());
    for (size_t i = 0; i < rows.size(); i++) {
        const polabi::GroupRow& r = rows[i];
        PyList_SET_ITEM(list, (Py_ssize_t)i,
                        Py_BuildValue("(LOLLLLL)", (long long)r.gid, r.anchored ? Py_True : Py_False,
                                      (long long)r.n_trees, (long long)r.n_fruits, (long long)r.n_cells,
                                      (long long)r.next_tree, (long long)r.next_fruit));
    }
    return list;
}

// Evasion counters, for reporting only; nothing here feeds a decision.
PyObject* PE_policy_metrics(PolicyEngineObject* self, PyObject*) {
    if (!self->pol) { PyErr_SetString(PyExc_RuntimeError, "policy_init first"); return nullptr; }
    int64_t flee = 0, face = 0, sprint = 0;
    self->pol->metrics(flee, face, sprint);
    return Py_BuildValue("{s:L,s:L,s:L}", "flee_ticks", (long long)flee, "face_ticks", (long long)face,
                         "sprint_ticks", (long long)sprint);
}

// THE FAST PATH. One Python call runs many ticks: decide, apply, step, repeat, with
// the GIL released and no Python object created or destroyed in the loop. Returns
// after the step at which sim_time >= stop_at - 1e-6 (for sampling) or when the run
// ends. A 3000 s game is one call, not 30000.
PyObject* PE_run_policy(PolicyEngineObject* self, PyObject* args) {
    double horizon, stop_at;
    int profile = 0;
    if (!PyArg_ParseTuple(args, "dd|p", &horizon, &stop_at, &profile)) return nullptr;
    if (!self->pol) { PyErr_SetString(PyExc_RuntimeError, "policy_init first"); return nullptr; }
    Engine* e = self->eng;
    polabi::IPolicy* pol = self->pol;
    std::vector<polabi::AState>& buf = *self->buf;
    long steps = 0; size_t peak = e->agents.size();
    // Nanosecond attribution, off by default so the fast path stays free of clock
    // reads. `iface` is everything spent handing state across the boundary.
    long long ns_iface = 0, ns_policy = 0, ns_engine = 0;
    Py_BEGIN_ALLOW_THREADS
    using clk = std::chrono::steady_clock;
    while (!e->agents.empty() && e->time < horizon) {
        if (profile) {
            auto t0 = clk::now();
            fill_states(e, buf);
            auto t1 = clk::now();
            const std::vector<polabi::Act>& acts = pol->call(buf.data(), buf.size(), e->time);
            auto t2 = clk::now();
            for (const polabi::Act& a : acts) {
                Engine::Action ea{a.aid, a.dist, true, a.direction, a.turn, a.spawn};
                e->agent_step(ea);
            }
            e->non_agent_step();
            auto t3 = clk::now();
            ns_iface += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            ns_policy += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            ns_engine += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        } else {
            fill_states(e, buf);
            const std::vector<polabi::Act>& acts = pol->call(buf.data(), buf.size(), e->time);
            for (const polabi::Act& a : acts) {
                Engine::Action ea{a.aid, a.dist, true, a.direction, a.turn, a.spawn};
                e->agent_step(ea);
            }
            e->non_agent_step();
        }
        steps++;
        if (e->agents.size() > peak) peak = e->agents.size();
        if (e->time >= stop_at - 1e-6) break;
    }
    Py_END_ALLOW_THREADS
    return Py_BuildValue("(nnLLL)", (Py_ssize_t)steps, (Py_ssize_t)peak, ns_iface, ns_policy, ns_engine);
}

// Engine-only loop with no policy: the same tick structure driven by zero actions.
// Used by the benchmark to separate engine time from policy time.
PyObject* PE_run_engine_only(PolicyEngineObject* self, PyObject* args) {
    double horizon;
    if (!PyArg_ParseTuple(args, "d", &horizon)) return nullptr;
    Engine* e = self->eng;
    long steps = 0;
    Py_BEGIN_ALLOW_THREADS
    while (!e->agents.empty() && e->time < horizon) { e->non_agent_step(); steps++; }
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(steps);
}

// Turn per-phase accounting inside the policy on or off (reporting only).
PyObject* PE_policy_profile(PolicyEngineObject* self, PyObject* arg) {
    if (!self->pol) { PyErr_SetString(PyExc_RuntimeError, "policy_init first"); return nullptr; }
    int on = PyObject_IsTrue(arg);
    if (on < 0) return nullptr;
    self->pol->set_profile(on != 0);
    Py_RETURN_NONE;
}

PyObject* PE_policy_phases(PolicyEngineObject* self, PyObject*) {
    if (!self->pol) { PyErr_SetString(PyExc_RuntimeError, "policy_init first"); return nullptr; }
    double sec[polabi::PH_N];
    self->pol->phases(sec);
    PyObject* d = PyDict_New();
    for (int i = 0; i < polabi::PH_N; i++) {
        PyObject* v = PyFloat_FromDouble(sec[i]);
        PyDict_SetItemString(d, polabi::PHASE_NAMES[i], v);
        Py_DECREF(v);
    }
    return d;
}

PyMethodDef policy_extra_methods[] = {
    {"policy_profile", (PyCFunction)PE_policy_profile, METH_O, "enable per-phase policy accounting"},
    {"policy_phases", (PyCFunction)PE_policy_phases, METH_NOARGS, "seconds per policy phase"},
    {"policy_init", (PyCFunction)PE_policy_init, METH_VARARGS,
     "policy_init(seed_key, config_dict): native orchard + evasion policy"},
    {"policy_act", (PyCFunction)PE_policy_act, METH_NOARGS,
     "native decisions for the current state: [(aid, dist, dir, turn, spawn)]"},
    {"policy_minds", (PyCFunction)PE_policy_minds, METH_NOARGS, "debug: native minds"},
    {"policy_groups", (PyCFunction)PE_policy_groups, METH_NOARGS, "debug: native group summaries"},
    {"policy_metrics", (PyCFunction)PE_policy_metrics, METH_NOARGS, "flee/face/sprint tick counts"},
    {"run_policy", (PyCFunction)PE_run_policy, METH_VARARGS,
     "run_policy(horizon, stop_at[, profile]) -> (steps, peak, ns_iface, ns_policy, ns_engine); "
     "whole tick loop in C++"},
    {"run_engine_only", (PyCFunction)PE_run_engine_only, METH_VARARGS,
     "run_engine_only(horizon) -> steps; non_agent_step only, for benchmark attribution"},
    {nullptr, nullptr, 0, nullptr}};

// The engine surface is taken from Engine_methods verbatim, so a method added to
// _engine.cpp later shows up here without this file being touched.
PyMethodDef* joined_methods() {
    size_t n = 0; while (Engine_methods[n].ml_name) n++;
    size_t m = 0; while (policy_extra_methods[m].ml_name) m++;
    PyMethodDef* out = new PyMethodDef[n + m + 1];
    for (size_t i = 0; i < n; i++) out[i] = Engine_methods[i];
    for (size_t i = 0; i < m; i++) out[n + i] = policy_extra_methods[i];
    out[n + m] = PyMethodDef{nullptr, nullptr, 0, nullptr};
    return out;
}

void PE_dealloc(PolicyEngineObject* self) {
    polabi::destroy_policy(self->pol);
    self->pol = nullptr;
    delete self->buf;
    self->buf = nullptr;
    Engine_dealloc((EngineObject*)self);
}

int PE_init(PolicyEngineObject* self, PyObject* args, PyObject* kwds) {
    polabi::destroy_policy(self->pol);
    self->pol = nullptr;
    if (!self->buf) self->buf = new std::vector<polabi::AState>();
    self->buf->clear();
    return Engine_init((EngineObject*)self, args, kwds);
}

PyTypeObject PolicyEngineType = {PyVarObject_HEAD_INIT(nullptr, 0)};

PyModuleDef policy_moduledef = {PyModuleDef_HEAD_INIT, "_policy",
                                "Native survival simulator engine with the native orchard evasion policy", -1,
                                module_methods};

}  // namespace

// Install the libm CPython's math module uses, so the policy's sin/cos/atan2/pow are
// bit-for-bit what the Python policy computes. On Windows that is the UCRT's, reached
// through ucrtbase.dll: the MinGW build of this extension would otherwise use
// libmingwex, which is 1 ULP off for some inputs. Everywhere else the compiler and
// CPython already share a libm, so the standard library is the right answer and the
// pm defaults are left in place. Returns false only if the UCRT lookup fails on
// Windows, which would silently reintroduce the divergence - so that is fatal.
bool install_policy_libm() {
#ifdef _WIN32
    HMODULE h = LoadLibraryA("ucrtbase.dll");
    if (!h) return false;
    auto s = (double (*)(double))(void*)GetProcAddress(h, "sin");
    auto c = (double (*)(double))(void*)GetProcAddress(h, "cos");
    auto a = (double (*)(double, double))(void*)GetProcAddress(h, "atan2");
    auto p = (double (*)(double, double))(void*)GetProcAddress(h, "pow");
    if (!s || !c || !a || !p) return false;
    polabi::pm::set_libm(s, c, a, p);
#endif
    return true;
}

PyMODINIT_FUNC PyInit__policy(void) {
    if (!install_policy_libm()) {
        PyErr_SetString(PyExc_ImportError,
                        "fastsim._policy: cannot resolve the UCRT math functions; the native "
                        "policy would not match math.sin/cos/atan2/pow bit for bit");
        return nullptr;
    }
    PolicyEngineType.tp_name = "_policy.Engine";
    PolicyEngineType.tp_basicsize = sizeof(PolicyEngineObject);
    PolicyEngineType.tp_flags = Py_TPFLAGS_DEFAULT;
    PolicyEngineType.tp_new = PyType_GenericNew;
    PolicyEngineType.tp_init = (initproc)PE_init;
    PolicyEngineType.tp_dealloc = (destructor)PE_dealloc;
    PolicyEngineType.tp_methods = joined_methods();
    if (PyType_Ready(&PolicyEngineType) < 0) return nullptr;
#define INTERN(var, s) var = PyUnicode_InternFromString(s)
    INTERN(s_type, "type"); INTERN(s_distance, "distance"); INTERN(s_angle, "angle"); INTERN(s_rel_dir, "rel_dir");
    INTERN(s_id, "id"); INTERN(s_coords, "coords"); INTERN(s_Fruit, "Fruit"); INTERN(s_Agent, "Agent");
    INTERN(s_Predator, "Predator"); INTERN(s_Tree, "Tree"); INTERN(s_Edge, "Edge");
    INTERN(s_agent_id, "agent_id"); INTERN(s_observations, "observations"); INTERN(s_energy, "energy");
    INTERN(s_biome, "biome"); INTERN(s_age, "age"); INTERN(s_speed, "speed"); INTERN(s_sprint_speed, "sprint_speed");
    INTERN(s_hearing_radius, "hearing_radius"); INTERN(s_vision_angle, "vision_angle"); INTERN(s_vision_range, "vision_range");
    INTERN(s_max_energy, "max_energy"); INTERN(s_score, "score"); INTERN(s_sim_time, "sim_time"); INTERN(s_num_agents, "num_agents");
    INTERN(s_move_distance, "move_distance"); INTERN(s_move_direction, "move_direction"); INTERN(s_turn_angle, "turn_angle");
    INTERN(s_spawn_agent, "spawn_agent");
    for (int i = 0; i < 5; i++) s_biome_names[i] = PyUnicode_InternFromString(BIOMES[i].name);
#undef INTERN
    PyObject* m = PyModule_Create(&policy_moduledef);
    if (!m) return nullptr;
    Py_INCREF(&PolicyEngineType);
    PyModule_AddObject(m, "Engine", (PyObject*)&PolicyEngineType);
    return m;
}
