// Reuse the simulator unchanged, adding an agent-free bulk scanner.
#define PyInit__engine PyInit__vendor_engine
#include "vendor_engine.cpp"
#undef PyInit__engine
#include <array>
#include <deque>
#include <numeric>
#include <memory>

namespace {
struct Sample { int tick; double x, y, heading, energy; bool resting; };
struct Window {
    int steps;
    double radius;
    std::deque<Sample> history;
    std::array<std::deque<std::pair<int, double>>, 4> extrema;
    Window(int n, double r) : steps(n), radius(r) {}
    bool add(Sample s) {
        if ((!history.empty() && s.tick != history.back().tick + 1) ||
            !std::isfinite(s.x) || !std::isfinite(s.y) ||
            !std::isfinite(s.heading) || !std::isfinite(s.energy))
            throw std::runtime_error("Invalid or nonconsecutive predator state");
        history.push_back(s);
        const int oldest = s.tick - steps;
        while (history.front().tick < oldest) history.pop_front();
        for (int i = 0; i < 4; ++i) {
            auto& q = extrema[i];
            double v = i < 2 ? s.x : s.y;
            while (!q.empty() && (i % 2 == 0 ? q.back().second >= v : q.back().second <= v)) q.pop_back();
            q.emplace_back(s.tick, v);
            while (q.front().first < oldest) q.pop_front();
        }
        if (history.size() < static_cast<size_t>(steps + 1)) return false;
        const auto& a = history.front();
        double dx = std::max(std::abs(extrema[0].front().second - a.x), std::abs(extrema[1].front().second - a.x));
        double dy = std::max(std::abs(extrema[2].front().second - a.y), std::abs(extrema[3].front().second - a.y));
        if (dx > radius || dy > radius) return false;
        if (dx * dx + dy * dy <= radius * radius) return true;
        for (const auto& p : history) {
            dx = p.x - a.x; dy = p.y - a.y;
            if (dx * dx + dy * dy > radius * radius) return false;
        }
        return true;
    }
};
struct Birth { int tick; double x, y, heading; bool overlap; };
struct Detail { Sample sample; PredatorDebug debug; };
struct Finding {
    size_t id; int biome; double size; std::deque<Sample> samples;
    std::deque<Detail> detail;
    int first_exit_tick=-1;
    double max_distance_after=0;
};
struct Totals {
    int rest=0, active=0, edge=0, wander=0, blocked=0, fallback=0, blocked_run=0, longest_blocked=0;
    double distance=0;
    void add(const PredatorDebug& d, const Sample& p) {
        if (d[D_MODE]<0) return;
        double moved=std::hypot(p.x-d[D_X],p.y-d[D_Y]); distance+=moved;
        if (d[D_MODE]==0) { rest++; blocked_run=0; return; }
        active++;
        if (d[D_MODE]==1) edge++;
        if (d[D_MODE]==2) wander++;
        if (d[D_ACCEPTED]<0) { blocked++; blocked_run++; longest_blocked=std::max(longest_blocked,blocked_run); }
        else blocked_run=0;
        if (d[D_ACCEPTED]>0) fallback++;
    }
};

PyObject* detail_python(const std::deque<Detail>& details) {
    auto* rows=PyList_New(details.size()); if (!rows) return nullptr;
    for (size_t i=0;i<details.size();i++) {
        const auto& f=details[i]; const auto& s=f.sample;
        auto* row=PyList_New(6+D_COUNT);
        if (!row) { Py_DECREF(rows); return nullptr; }
        PyList_SET_ITEM(row,0,PyLong_FromLong(s.tick));
        const double post[5]={s.x,s.y,s.heading,s.energy,static_cast<double>(s.resting)};
        for(int k=0;k<5;k++) PyList_SET_ITEM(row,1+k,PyFloat_FromDouble(post[k]));
        for(int k=0;k<D_COUNT;k++) PyList_SET_ITEM(row,6+k,PyFloat_FromDouble(f.debug[k]));
        PyList_SET_ITEM(rows,i,row);
    }
    return rows;
}

PyObject* samples_python(const std::deque<Sample>& samples) {
    PyObject* rows = PyList_New(samples.size());
    if (!rows) return nullptr;
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& s = samples[i];
        auto* row = Py_BuildValue("(iddddN)", s.tick, s.x, s.y, s.heading, s.energy, PyBool_FromLong(s.resting));
        if (!row) { Py_DECREF(rows); return nullptr; }
        PyList_SET_ITEM(rows, i, row);
    }
    return rows;
}

Engine* require_engine(PyObject* object) {
    if (!PyObject_TypeCheck(object, &EngineType)) {
        PyErr_SetString(PyExc_TypeError, "Expected a native Engine"); return nullptr;
    }
    auto* e = reinterpret_cast<EngineObject*>(object)->eng;
    if (!e || !e->agents.empty()) {
        PyErr_SetString(PyExc_ValueError, "Scanner requires an initialized engine with zero agents"); return nullptr;
    }
    return e;
}

PyObject* ensure_predators(PyObject*, PyObject* args) {
    PyObject* object; int target;
    if (!PyArg_ParseTuple(args, "Oi", &object, &target)) return nullptr;
    auto* e = require_engine(object); if (!e) return nullptr;
    if (target < 1 || target > 100000 || !e->predators_enabled) {
        PyErr_SetString(PyExc_ValueError, "Invalid predator population"); return nullptr;
    }
    int attempts = 0;
    while (e->predators.size() < static_cast<size_t>(target)) {
        if (++attempts > target * 1000) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to place requested predators"); return nullptr;
        }
        e->spawn_predator();
    }
    return PyLong_FromLong(attempts);
}

PyObject* run_scan(PyObject*, PyObject* args) {
    PyObject* object; int ticks, steps, detailed=0, pre_steps=300; double radius;
    if (!PyArg_ParseTuple(args, "Oiid|pi", &object, &ticks, &steps, &radius, &detailed, &pre_steps)) return nullptr;
    auto* e = require_engine(object); if (!e) return nullptr;
    if (ticks < steps || ticks == INT_MAX || steps < 1 || steps == INT_MAX || pre_steps<0 || pre_steps>10000 ||
        steps>INT_MAX-pre_steps-1 || !std::isfinite(radius) || radius <= 0 || e->time != 0) {
        PyErr_SetString(PyExc_ValueError, "Positive interval/radius and a fresh engine required"); return nullptr;
    }
    std::vector<Birth> births;
    std::vector<std::unique_ptr<Window>> windows;
    std::vector<Finding> findings;
    std::vector<std::deque<Detail>> recent;
    std::vector<Totals> totals;
    std::vector<int> finding_index;
    e->predator_debug_enabled = detailed != 0;
    std::vector<int32_t> obstacles(e->obstacles.size());
    std::iota(obstacles.begin(), obstacles.end(), 0);
    std::string error;
    // No Python objects/callbacks or per-tick GIL transitions in this loop.
    Py_BEGIN_ALLOW_THREADS
    try {
        for (int tick = 0; tick <= ticks; ++tick) {
            if (tick) e->non_agent_step();
            while (births.size() < e->predators.size()) {
                const auto& p = e->predators[births.size()];
                births.push_back({tick, p.x, p.y, p.direction, e->in_obstacle(p.x, p.y, p.size, obstacles)});
                windows.push_back(std::make_unique<Window>(steps, radius));
                recent.emplace_back(); totals.emplace_back(); finding_index.push_back(-1);
            }
            for (size_t id = 0; id < windows.size(); ++id) {
                const auto& p = e->predators[id];
                Sample sample{tick,p.x,p.y,p.direction,p.energy,p.resting};
                if (detailed) {
                    PredatorDebug debug{};
                    debug[D_MODE]=-1; debug[D_ACCEPTED]=-1; debug[D_FIRST_BLOCKER]=-1;
                    debug[D_X]=p.x; debug[D_Y]=p.y; debug[D_HEADING]=p.direction;
                    debug[D_ENERGY]=p.energy; debug[D_REST]=p.resting; debug[D_BIOME]=e->biome_at(p.x,p.y);
                    if (tick && id<e->predator_debug.size()) debug=e->predator_debug[id];
                    totals[id].add(debug,sample);
                    if (windows[id]) {
                        recent[id].push_back({sample,debug});
                        if (recent[id].size()>static_cast<size_t>(steps+pre_steps+1)) recent[id].pop_front();
                    }
                }
                if (!windows[id]) {
                    auto& f=findings[finding_index[id]];
                    const auto& a=f.samples.front();
                    double distance=std::hypot(p.x-a.x,p.y-a.y);
                    f.max_distance_after=std::max(f.max_distance_after,distance);
                    if (distance>radius && f.first_exit_tick<0) f.first_exit_tick=tick;
                    continue;
                }
                auto& w = *windows[id];
                if (w.add(sample)) {
                    finding_index[id]=static_cast<int>(findings.size());
                    findings.push_back({id, e->biome_at(p.x, p.y), p.size, std::move(w.history)});
                    if(detailed) findings.back().detail=std::move(recent[id]);
                    windows[id].reset();
                }
            }
        }
    } catch (const std::exception& exc) { error = exc.what(); }
    Py_END_ALLOW_THREADS
    if (!error.empty()) { PyErr_SetString(PyExc_RuntimeError, error.c_str()); return nullptr; }
    PyObject* catalog = PyList_New(births.size());
    PyObject* events = PyList_New(findings.size());
    if (!catalog || !events) { Py_XDECREF(catalog); Py_XDECREF(events); return nullptr; }
    for (size_t id = 0; id < births.size(); ++id) {
        const auto& b = births[id];
        PyList_SET_ITEM(catalog, id, Py_BuildValue("(idddN)", b.tick, b.x, b.y, b.heading, PyBool_FromLong(b.overlap)));
    }
    for (size_t i = 0; i < findings.size(); ++i) {
        const auto& f = findings[i];
        PyList_SET_ITEM(events, i, Py_BuildValue("(nsdN)", static_cast<Py_ssize_t>(f.id), BIOMES[f.biome].name,
                                              f.size, samples_python(f.samples)));
    }
    PyObject* debug_rows=PyList_New(findings.size());
    for (size_t i=0;i<findings.size();i++) {
        const auto& f=findings[i]; const auto& p=e->predators[f.id];
        PyList_SET_ITEM(debug_rows,i,Py_BuildValue("(nNiddd)",static_cast<Py_ssize_t>(f.id),
            detail_python(f.detail),f.first_exit_tick,f.max_distance_after,p.x,p.y));
    }
    PyObject* population=PyList_New(totals.size());
    for(size_t i=0;i<totals.size();i++) {
        const auto& t=totals[i];
        PyList_SET_ITEM(population,i,Py_BuildValue("(iiiiiiid)",t.rest,t.active,t.edge,t.wander,
            t.blocked,t.fallback,t.longest_blocked,t.distance));
    }
    // One unflagged predator's final window per game is a trace-level control.
    PyObject* control=Py_None; Py_INCREF(control);
    if(detailed) for(size_t i=0;i<windows.size();i++) if(windows[i]) {
        Py_DECREF(control); control=Py_BuildValue("(nN)",static_cast<Py_ssize_t>(i),detail_python(recent[i])); break;
    }
    return Py_BuildValue("{s:N,s:N,s:N,s:N,s:N}", "births", catalog, "events", events,
        "diagnostics",debug_rows,"population",population,"control",control);
}

// Replay a seed to a checkpoint, then branch identical worlds for causal probes.
// This is a separate evaluator API; the batch scanner never invokes it.
PyObject* probe_pose(PyObject*, PyObject* args) {
    PyObject *object,*input; int pid, checkpoint, duration;
    if(!PyArg_ParseTuple(args,"OiiiO",&object,&pid,&checkpoint,&duration,&input)) return nullptr;
    auto* base=require_engine(object); if(!base) return nullptr;
    if(base->time!=0 || pid<0 || checkpoint<0 || duration<1 || duration>100000 || checkpoint>100000) {
        PyErr_SetString(PyExc_ValueError,"Fresh engine, valid predator and bounded times required");return nullptr;
    }
    auto* seq=PySequence_Fast(input,"Expected (dx,dy,heading_delta,remove_obstacle) variants"); if(!seq)return nullptr;
    struct Variant { double dx,dy,heading; int remove; int exit=-1; double max_distance=0; bool overlap=false;
        std::deque<Sample> path; Totals totals; };
    std::vector<Variant> variants;
    for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(seq);i++) {
        Variant v;
        if(!PyArg_ParseTuple(PySequence_Fast_GET_ITEM(seq,i),"dddi",&v.dx,&v.dy,&v.heading,&v.remove)) {
            Py_DECREF(seq);return nullptr;
        }
        if(!std::isfinite(v.dx)||!std::isfinite(v.dy)||!std::isfinite(v.heading)||
            v.remove < -1 || v.remove>=static_cast<int>(base->obstacles.size())) {
            Py_DECREF(seq);PyErr_SetString(PyExc_ValueError,"Invalid intervention");return nullptr;
        }
        variants.push_back(v);
    }
    Py_DECREF(seq);
    std::string error;
    Py_BEGIN_ALLOW_THREADS
    try {
        for(int t=0;t<checkpoint;t++) base->non_agent_step();
        if(pid>=static_cast<int>(base->predators.size())) throw std::runtime_error("Predator not born at checkpoint");
        for(auto& v:variants) {
            Engine e=*base;
            e.active_predator_debug=nullptr; e.predator_debug_enabled=true;
            auto& p=e.predators[pid];p.x+=v.dx;p.y+=v.dy;p.direction+=v.heading;
            if(p.x<10||p.x>e.W-10||p.y<10||p.y>e.H-10)throw std::runtime_error("Perturbed pose outside bounds");
            const double ax=p.x,ay=p.y;
            e.predators_dirty=true;
            if(v.remove>=0) {
                e.obstacles.erase(e.obstacles.begin()+v.remove);
                for(auto& g:e.grid_edges)g.clear();
                for(auto& g:e.grid_obstacles)g.clear();
                std::fill(e.local_cache_ready.begin(),e.local_cache_ready.end(),false);
                e.cell_edge_cand.clear();e.radius_ids.clear();e.build_edges();
            }
            std::vector<int32_t> obs(e.obstacles.size());std::iota(obs.begin(),obs.end(),0);
            v.overlap=e.in_obstacle(p.x,p.y,p.size,obs);
            for(int t=0;t<=duration;t++) {
                if(t)e.non_agent_step();
                const auto& q=e.predators[pid];
                Sample s{checkpoint+t,q.x,q.y,q.direction,q.energy,q.resting};
                if(t)v.totals.add(e.predator_debug[pid],s);
                if(t%10==0 || t==duration)v.path.push_back(s);
                double distance=std::hypot(q.x-ax,q.y-ay);
                v.max_distance=std::max(v.max_distance,distance);
                if(distance>15 && v.exit<0)v.exit=t;
            }
        }
    }catch(const std::exception& exc){error=exc.what();}
    Py_END_ALLOW_THREADS
    if(!error.empty()){PyErr_SetString(PyExc_RuntimeError,error.c_str());return nullptr;}
    auto* results=PyList_New(variants.size());if(!results)return nullptr;
    for(size_t i=0;i<variants.size();i++) {
        const auto& v=variants[i];
        PyList_SET_ITEM(results,i,Py_BuildValue("{s:i,s:d,s:O,s:N,s:i,s:i}",
            "first_exit_tick",v.exit,"max_distance",v.max_distance,"initial_overlap",v.overlap?Py_True:Py_False,
            "path",samples_python(v.path),"active_ticks",v.totals.active,"blocked_ticks",v.totals.blocked));
    }
    return results;
}

// Exercise the identical detector on arbitrary paths, independently of physics.
PyObject* detect_path(PyObject*, PyObject* args) {
    PyObject* input; int steps; double radius;
    if (!PyArg_ParseTuple(args, "Oid", &input, &steps, &radius)) return nullptr;
    if (steps < 1 || steps == INT_MAX || !std::isfinite(radius) || radius <= 0) {
        PyErr_SetString(PyExc_ValueError, "Positive interval and radius required"); return nullptr;
    }
    auto* seq = PySequence_Fast(input, "Expected coordinate pairs"); if (!seq) return nullptr;
    auto n = PySequence_Fast_GET_SIZE(seq);
    auto* result = PyList_New(n); if (!result) { Py_DECREF(seq); return nullptr; }
    Window window(steps, radius);
    try {
        for (Py_ssize_t i = 0; i < n; ++i) {
            double x, y;
            if (!PyArg_ParseTuple(PySequence_Fast_GET_ITEM(seq, i), "dd", &x, &y)) {
                Py_DECREF(seq); Py_DECREF(result); return nullptr;
            }
            PyList_SET_ITEM(result, i, PyBool_FromLong(window.add({static_cast<int>(i), x, y, 0, 0, false})));
        }
    } catch (const std::exception& exc) {
        Py_DECREF(seq); Py_DECREF(result); PyErr_SetString(PyExc_ValueError, exc.what()); return nullptr;
    }
    Py_DECREF(seq); return result;
}
PyMethodDef scan_methods[] = {
    {"probe_pose", probe_pose, METH_VARARGS, "Replay then branch independent heading/position/geometry interventions."},
    {"ensure_predators", ensure_predators, METH_VARARGS, "Retry native spawning to reach a population."},
    {"run_scan", run_scan, METH_VARARGS, "Run every simulation tick and confinement check in C++."},
    {"detect_path", detect_path, METH_VARARGS, "Test rolling confinement on coordinate pairs."},
    {nullptr, nullptr, 0, nullptr}};
}
PyMODINIT_FUNC PyInit__stuck(void) {
    auto* module = PyInit__vendor_engine();
    if (module && PyModule_AddFunctions(module, scan_methods) < 0) { Py_DECREF(module); return nullptr; }
    return module;
}
