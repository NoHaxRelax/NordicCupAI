// fastsim._mirror - the fastsim engine plus Engine.clone(), for synchronized shadow
// models and branch-and-replay search.
//
// Same arrangement as _policy.cpp and for the same reason: fastsim/_engine.cpp is a
// provenance-pinned artefact (fastsim/build-info.json records its source_sha256), so it
// is left byte-for-byte alone and #include'd here verbatim. This translation unit gets
// its own private copy of the engine and compiles to its own _mirror<EXT_SUFFIX>.
// `import fastsim` keeps using _engine exactly as before.
//
// Why a clone is all that is needed: class Engine (_engine.cpp:539) has only value
// members - std::vector, std::unordered_map and PySetEmu{std::vector<Entry>,...} - with
// no raw pointers, no references, and no user-declared copy constructor, copy assignment
// or destructor. The implicitly generated copy constructor is therefore already a correct
// deep copy, including the 624-word MT state. There was simply never a binding for it.
//
// Rather than edit the included Engine_methods[] table, clone is attached to the ready
// type as a method descriptor in the module init, which leaves the included source
// untouched.
//
// NOTE ON COST: a clone copies the immutable map data too - the 1.92 MB biome raster,
// obstacles, edges, grid_edges, grid_obstacles, corner_cells, edge_cells and the
// local_* caches. That is correct but wasteful; every one of those is fixed after
// construction. Sharing them behind a shared_ptr would cut a clone to the RNG state plus
// the entity vectors. Measure before optimising - see mirror/README.md.

#define PyInit__engine PyInit__engine_not_used_here
#include "_engine.cpp"
#undef PyInit__engine

namespace {

// Deep copy via Engine's implicit copy constructor. The new object is the same Python
// type as the receiver, so clones clone.
PyObject* Engine_clone(EngineObject* self, PyObject*) {
    if (!self->eng) { PyErr_SetString(PyExc_ValueError, "engine not initialised"); return nullptr; }
    PyTypeObject* type = Py_TYPE((PyObject*)self);
    EngineObject* obj = (EngineObject*)type->tp_alloc(type, 0);
    if (!obj) return nullptr;
    obj->eng = nullptr;
    Engine* copy = nullptr;
    Py_BEGIN_ALLOW_THREADS
    copy = new Engine(*self->eng);
    Py_END_ALLOW_THREADS
    obj->eng = copy;
    return (PyObject*)obj;
}

PyMethodDef clone_def = {"clone", (PyCFunction)Engine_clone, METH_NOARGS,
                         "clone() -> Engine; deep copy including the exact RNG state"};

// ---------------------------------------------------------------- snapshot/restore
//
// clone() copies the whole engine, including ~2 MB of map data that is fixed at
// construction, which makes it far too expensive to use as a search primitive. Search
// does not need a second engine: it needs to rewind one. Everything below is the
// genuinely dynamic state; everything omitted is either immutable after construction
// (biome, obstacles, edges, the chunk grids over them, corner/edge cells) or a pure
// cache that refresh_grids() rebuilds from the dirty flags (grid_agents/fruits/trees/
// predators, uc_*, key_to_index_*), or a static-map-keyed cache that stays valid across
// a rewind and is better kept warm (cell_edge_cand, radius_ids).
struct Snapshot {
    PyRandom rng;
    double score, time;
    int64_t next_agent_id, next_fruit_id, next_serial;
    std::vector<Creature> agents, predators;
    std::vector<Fruit> fruits;
    std::vector<Tree> trees;
    std::unordered_map<int64_t, std::vector<Obs>> agent_observations;
    size_t n_events;
    bool diagnostics_enabled;
};

void snapshot_save(const Engine& e, Snapshot& s) {
    s.rng = e.rng;
    s.score = e.score; s.time = e.time;
    s.next_agent_id = e.next_agent_id; s.next_fruit_id = e.next_fruit_id; s.next_serial = e.next_serial;
    s.agents = e.agents; s.predators = e.predators;
    s.fruits = e.fruits; s.trees = e.trees;
    s.agent_observations = e.agent_observations;
    s.n_events = e.events.size();
    s.diagnostics_enabled = e.diagnostics_enabled;
}

void snapshot_load(Engine& e, const Snapshot& s) {
    e.rng = s.rng;
    e.score = s.score; e.time = s.time;
    e.next_agent_id = s.next_agent_id; e.next_fruit_id = s.next_fruit_id; e.next_serial = s.next_serial;
    e.agents = s.agents; e.predators = s.predators;
    e.fruits = s.fruits; e.trees = s.trees;
    e.events.resize(s.n_events);
    e.diagnostics_enabled = s.diagnostics_enabled;
    e.diagnostics.clear();
    // Observations MUST be carried, not cleared. They are written in the agents loop
    // (_engine.cpp:1423) and read by build_state (:1662), so they are what the policy sees
    // on the FIRST tick after a restore. Clearing them leaves that tick blind, which does
    // not change the engine's own evolution -- an exactness test will still pass -- but it
    // silently feeds a search's branches a different problem from the one being solved.
    e.agent_observations = s.agent_observations;
    // Force refresh_grids() to rebuild the grids, the union caches and the key indices.
    e.agents_dirty = e.fruits_dirty = e.trees_dirty = e.predators_dirty = true;
}

struct SnapshotObject {
    PyObject_HEAD
    Snapshot* snap;
};

PyTypeObject SnapshotType = {PyVarObject_HEAD_INIT(nullptr, 0)};

void Snapshot_dealloc(SnapshotObject* self) {
    delete self->snap;
    Py_TYPE((PyObject*)self)->tp_free((PyObject*)self);
}

PyObject* Engine_snapshot(EngineObject* self, PyObject*) {
    if (!self->eng) { PyErr_SetString(PyExc_ValueError, "engine not initialised"); return nullptr; }
    SnapshotObject* obj = (SnapshotObject*)SnapshotType.tp_alloc(&SnapshotType, 0);
    if (!obj) return nullptr;
    obj->snap = new Snapshot();
    snapshot_save(*self->eng, *obj->snap);
    return (PyObject*)obj;
}

PyObject* Engine_restore(EngineObject* self, PyObject* arg) {
    if (!self->eng) { PyErr_SetString(PyExc_ValueError, "engine not initialised"); return nullptr; }
    if (!PyObject_TypeCheck(arg, &SnapshotType)) {
        PyErr_SetString(PyExc_TypeError, "restore() needs a Snapshot from Engine.snapshot()");
        return nullptr;
    }
    snapshot_load(*self->eng, *((SnapshotObject*)arg)->snap);
    Py_RETURN_NONE;
}

PyMethodDef snapshot_def = {"snapshot", (PyCFunction)Engine_snapshot, METH_NOARGS,
                            "snapshot() -> Snapshot; capture the dynamic state cheaply"};
PyMethodDef restore_def = {"restore", (PyCFunction)Engine_restore, METH_O,
                           "restore(snapshot); rewind this engine to a snapshot"};

PyModuleDef mirror_moduledef = {PyModuleDef_HEAD_INIT, "_mirror",
                                "fastsim engine with Engine.clone() for shadow models", -1,
                                module_methods};

}  // namespace

PyMODINIT_FUNC PyInit__mirror(void) {
    // Run the included module's init purely for its side effects: it interns every
    // dictionary key the state/step paths use and calls PyType_Ready on EngineType.
    // The module object it returns is this TU's private "_engine" and is discarded.
    PyObject* base = PyInit__engine_not_used_here();
    if (!base) return nullptr;
    Py_DECREF(base);

    SnapshotType.tp_name = "_mirror.Snapshot";
    SnapshotType.tp_basicsize = sizeof(SnapshotObject);
    SnapshotType.tp_flags = Py_TPFLAGS_DEFAULT;
    SnapshotType.tp_new = PyType_GenericNew;
    SnapshotType.tp_dealloc = (destructor)Snapshot_dealloc;
    if (PyType_Ready(&SnapshotType) < 0) return nullptr;

    for (PyMethodDef* def : {&clone_def, &snapshot_def, &restore_def}) {
        PyObject* descr = PyDescr_NewMethod(&EngineType, def);
        if (!descr) return nullptr;
        if (PyDict_SetItemString(EngineType.tp_dict, def->ml_name, descr) < 0) {
            Py_DECREF(descr); return nullptr;
        }
        Py_DECREF(descr);
    }
    PyType_Modified(&EngineType);

    PyObject* m = PyModule_Create(&mirror_moduledef);
    if (!m) return nullptr;
    Py_INCREF(&EngineType);
    PyModule_AddObject(m, "Engine", (PyObject*)&EngineType);
    Py_INCREF(&SnapshotType);
    PyModule_AddObject(m, "Snapshot", (PyObject*)&SnapshotType);
    return m;
}
