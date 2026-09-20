// Engine bridge. Harvest decisions compile separately against the public ABI.
#define PyInit__policy PyInit__harvest_base
#include "_policy.cpp"
#undef PyInit__policy
#include "../models/native_harvest_iface.hpp"
#include "../scripts/harvest_bo_objective.hpp"
#include <exception>

static PyObject* run_harvest_native(PolicyEngineObject* self,PyObject* args) {
    double horizon=3000,engage=45;int min_free=6,max_ticks=2;
    if(!PyArg_ParseTuple(args,"|ddii",&horizon,&engage,&min_free,&max_ticks))return nullptr;
    if(!self->pol){PyErr_SetString(PyExc_RuntimeError,"policy_init required");return nullptr;}
    if(!std::isfinite(horizon)||horizon<=0||horizon>3000||!std::isfinite(engage)||engage<=0||engage>100||min_free<1||max_ticks<1||max_ticks>3){PyErr_SetString(PyExc_ValueError,"invalid harvest configuration");return nullptr;}
    auto* e=self->eng;if(e->time!=0){PyErr_SetString(PyExc_ValueError,"run_harvest requires a fresh game");return nullptr;}
    std::unique_ptr<native_harvest_api::Handle,decltype(&native_harvest_api::destroy)> hc(native_harvest_api::create(horizon,engage,min_free,max_ticks),native_harvest_api::destroy);
    std::exception_ptr failure;long long steps=0;
    Py_BEGIN_ALLOW_THREADS
    try {
        e->non_agent_step();
        while(!e->agents.empty()&&e->time<horizon){
            fill_states(e,*self->buf);
            auto& filtered=native_harvest_api::observe(hc.get(),*self->buf,e->time,e->score);
            auto& native=self->pol->call(filtered.data(),filtered.size(),e->time);
            auto out=native_harvest_api::actions(hc.get(),*self->buf,e->time,native);
            for(auto& a:out)e->agent_step(Engine::Action{a.aid,a.dist,true,a.direction,a.turn,a.spawn});
            e->non_agent_step();e->events.clear();++steps;
        }
        fill_states(e,*self->buf);native_harvest_api::observe(hc.get(),*self->buf,e->time,e->score);
    }catch(...){failure=std::current_exception();}
    Py_END_ALLOW_THREADS
    if(failure){try{std::rethrow_exception(failure);}catch(const std::exception& ex){PyErr_SetString(PyExc_RuntimeError,ex.what());}catch(...){PyErr_SetString(PyExc_RuntimeError,"native harvest failed");}return nullptr;}
    auto m=native_harvest_api::metrics(hc.get());
    return Py_BuildValue("{s:L,s:d,s:d,s:L,s:L,s:L,s:L,s:L,s:L,s:L}","steps",steps,"score",e->score,"sim_time",e->time,"attempts",(long long)m.attempts,"confirmed",(long long)m.confirmed,"failed",(long long)m.failed,"retries",(long long)m.retries,"sacrifices",(long long)m.sacrifices,"ignored",(long long)m.ignored,"dupes",(long long)m.dupes);
}
static PyObject* bo_summary(PyObject*,PyObject* arg) {
    PyObject* seq=PySequence_Fast(arg,"scores must be a sequence");if(!seq)return nullptr;
    std::vector<double> scores;auto n=PySequence_Fast_GET_SIZE(seq);scores.reserve(n);
    for(Py_ssize_t i=0;i<n;++i){double v=PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq,i));if(PyErr_Occurred()){Py_DECREF(seq);return nullptr;}scores.push_back(v);}Py_DECREF(seq);
    try {auto r=harvest_bo::summarize(std::move(scores));return Py_BuildValue("{s:d,s:d,s:d,s:d,s:d}","objective",r.objective(),"worst10",r.worst10,"worst100",r.worst100,"mean",r.mean,"minimum",r.minimum);}
    catch(const std::exception& ex){PyErr_SetString(PyExc_ValueError,ex.what());return nullptr;}
}
static PyMethodDef harvest_method={"run_harvest",(PyCFunction)run_harvest_native,METH_VARARGS,"Complete native harvest game (horizon, engage, min_free, max_ticks)."};
static void capsule_delete(PyObject* obj){native_harvest_api::destroy((native_harvest_api::Handle*)PyCapsule_GetPointer(obj,"harvest.controller"));}
static PyObject* debug_controller(PyObject*,PyObject*) {
    return PyCapsule_New(native_harvest_api::create(3000,45,6,2),"harvest.controller",capsule_delete);
}
static PyObject* debug_decide(PolicyEngineObject* self,PyObject* capsule) {
    auto* h=(native_harvest_api::Handle*)PyCapsule_GetPointer(capsule,"harvest.controller");if(!h)return nullptr;
    if(!self->pol){PyErr_SetString(PyExc_RuntimeError,"policy_init required");return nullptr;}
    fill_states(self->eng,*self->buf);
    auto& filtered=native_harvest_api::observe(h,*self->buf,self->eng->time,self->eng->score);
    auto& native=self->pol->call(filtered.data(),filtered.size(),self->eng->time);
    return acts_to_py(native_harvest_api::actions(h,*self->buf,self->eng->time,native));
}
static PyMethodDef debug_method={"harvest_decide",(PyCFunction)debug_decide,METH_O,"Public-state lockstep verification."};
static PyMethodDef summary_methods[]={{"bo_summary",bo_summary,METH_O,"Complete 1000-seed combined objective."},{"debug_controller",debug_controller,METH_NOARGS,"Create lockstep controller."},{nullptr,nullptr,0,nullptr}};
PyMODINIT_FUNC PyInit__harvest(void) {
    policy_moduledef.m_name="_harvest";
    PyObject* m=PyInit__harvest_base();if(!m)return nullptr;
    PyObject* descriptor=PyDescr_NewMethod(&PolicyEngineType,&harvest_method);
    if(!descriptor||PyDict_SetItemString(PolicyEngineType.tp_dict,"run_harvest",descriptor)<0){Py_XDECREF(descriptor);Py_DECREF(m);return nullptr;}
    Py_DECREF(descriptor);PyType_Modified(&PolicyEngineType);
    descriptor=PyDescr_NewMethod(&PolicyEngineType,&debug_method);
    if(!descriptor||PyDict_SetItemString(PolicyEngineType.tp_dict,"harvest_decide",descriptor)<0){Py_XDECREF(descriptor);Py_DECREF(m);return nullptr;}
    Py_DECREF(descriptor);PyType_Modified(&PolicyEngineType);
    if(PyModule_AddFunctions(m,summary_methods)<0){Py_DECREF(m);return nullptr;}
    PyModule_AddStringConstant(m,"HARVEST_IMPLEMENTATION","native-public-observation-v9");return m;
}
