'use strict';
const $ = id => document.getElementById(id);
let report = null, lastCases = '', lastReviews = '', lastEvents = '', selectedCase = '';
const fmt = (v, digits=1) => typeof v === 'number' && Number.isFinite(v) ? v.toLocaleString(undefined, {maximumFractionDigits:digits, minimumFractionDigits:digits}) : '—';
const money = v => '$' + fmt(v, 2);
const duration = v => {if (v == null) return '—'; const n=Math.max(0,Math.floor(v)); return `${Math.floor(n/3600)}h ${Math.floor(n/60)%60}m`;};
const date = v => v ? new Date(v*1000).toLocaleString() : '—';
function node(tag, text, cls) {const e=document.createElement(tag); if(text != null)e.textContent=String(text);if(cls)e.className=cls;return e;}
function set(id,value){$(id).textContent=value;}
function empty(parent,text){parent.replaceChildren(node('p',text,'empty'));}
function facts(parent, rows){parent.replaceChildren();for(const [k,v] of rows){parent.append(node('dt',k),node('dd',v));}}
function width(id, value){$(id).style.width=`${Math.min(100,Math.max(0,value || 0))}%`;}
function options(select, entries){const current=select.value;const signature=JSON.stringify(entries);if(select.dataset.signature===signature)return;select.replaceChildren(...entries.map(([value,label])=>{const o=node('option',label);o.value=value;return o;}));select.dataset.signature=signature;if(entries.some(e=>e[0]===current))select.value=current;}
function render(data){
  report=data;const s=data.state,b=data.budget,limits=data.limits,now=Date.now()/1000;
  const fleet=data.fleet || {tasks:[],estimated_compute_usd:0,reserved_usd:0};
  const committedAuxiliary=b.auxiliary_reserved_usd||0;
  const primarySpend=(b.estimated_compute_usd||0)+(b.external_spend_usd||0)-committedAuxiliary, auxiliarySpend=fleet.estimated_compute_usd||0;
  const separateFleet=!b.fleet_accounting_included && (committedAuxiliary>0 || !data.historical_spend_includes_fleet);
  const spend=primarySpend+(separateFleet?auxiliarySpend:0), auxiliaryReserve=separateFleet?Math.max(0,(committedAuxiliary||fleet.reserved_usd||0)-auxiliarySpend):0;
  set('campaign',data.campaign);set('generation',`${s.major || 1}.${s.sub || 1}`);set('stage',`${s.status} · ${s.stage}`);
  set('spend',money(spend));set('budget-note',b.expandable_spending?`${data.dispatch?.workers.length || b.deployed_pods || 4} Pods · expanded spending authorized`:`of ${money(limits.total_usd)} · estimated`);
  set('survival',data.best_summary ? `${fmt(data.best_summary.mean_survival)}s` : 'Pending');
  set('best-note',data.best_summary ? `${data.best.label} · ${data.best_summary.seeds} Python maps` : 'Awaiting initial comparison');
  set('best-updated',data.best_summary?.evaluated_at ? `Last comparison ${date(data.best_summary.evaluated_at)}` : '');
  const latestDecision=[...(data.rounds || [])].reverse().find(r=>r.winner?.id===data.best?.id);
  set('best-status',latestDecision ? `${latestDecision.name}: ${latestDecision.improved?'new policy accepted':'current best retained'}.` : 'Updates after a completed Python comparison.');
  set('completed',fmt(data.completed_cases,0));set('active-count',`${data.active.length} active on main Pod · additional Pods below`);
  if(data.dispatch){const busy=data.dispatch.workers.reduce((n,w)=>n+(w.busy_slots||0),0);set('active-count',`${busy} occupied game slots across ${data.dispatch.workers.length} Pods`);}
  set('review-count',`LLM reviews ${s.agent_calls || 0} / ${data.max_agent_calls ?? '—'}`);set('status',s.status);set('budget-total',money(limits.total_usd));
  width('spent-bar',spend/limits.total_usd*100);width('agent-bar',((b.agent_reserved_usd||0)+auxiliaryReserve)/limits.total_usd*100);width('reserve-bar',(limits.reserve_usd||0)/limits.total_usd*100);
  facts($('budget-detail'),[['Main Pod compute + setup',money(primarySpend)],['Additional Pods compute',money(auxiliarySpend)],['Additional compute reserved',money(auxiliaryReserve)],['LLM allowance reserved',money(b.agent_reserved_usd||0)],['Safety reserve',money(limits.reserve_usd)],['Unallocated estimate',money(Math.max(0,limits.total_usd-spend-auxiliaryReserve-(b.agent_reserved_usd||0)-(limits.reserve_usd||0)))]]);
  if(b.expandable_spending){set('budget-total','Expanded');facts($('budget-detail'),[['Fleet + earlier spending estimate',money(spend)],['Fleet compute rate',`${money(data.dispatch?.compute_hourly_usd || b.compute_hourly_usd || 3.84)}/hour + storage`],['Former $40 target','No longer stops the run'],['Normal completion','Generation 8 and final assessment'],['Emergency runtime limit',`${limits.max_hours} hours from main Pod creation`]]);}
  $('fleet-panel').hidden=!fleet.tasks.length;$('fleet').replaceChildren();
  set('fleet-budget',`${money(auxiliarySpend)} estimated · ${money(fleet.reserved_usd)} allocation`);
  if(data.dispatch){set('fleet-budget',`${data.dispatch.pending_or_active} shared jobs pending or running · main evaluations have priority`);for(const w of data.dispatch.workers){if(!w.id)continue;const card=node('article',null,'run');card.append(node('h3',`Pod ${w.id}`),node('div',`${fmt(w.cpu_percent,0)}% CPU · ${w.busy_slots} occupied / ${w.slots} target slots`,'run-detail'),node('div',`${w.active} shared jobs · ${w.legacy_active} earlier study games`,'run-detail'),node('div',now-w.updated_at>60?'Worker heartbeat stale':'Worker heartbeat current','run-detail'));$('fleet').append(card);}}
  for(const task of fleet.tasks){const card=node('article',null,'run');card.append(node('h3',task.title),node('div',`${task.status || 'starting'} · ${task.stage || 'preparing'}`,'run-detail'),node('div',`${task.completed_candidates}/${task.planned_candidates} candidates · ${task.active_cases.length} active runs`,'run-detail'));if(task.validation_total)card.append(node('div',`Python validation: ${task.validation_completed}/${task.validation_total}`,'run-detail'));card.append(node('div',`Guard ${task.guard_armed?'armed':'pending'} · cutoff ${date(task.deadline)}`,'run-detail'),node('div',`${money(task.estimated_compute_usd || 0)} estimated · ${task.pod_id || ''}`,'run-detail'));if(task.error||task.stop_reason)card.append(node('p',task.error||task.stop_reason,'muted'));for(const c of task.active_cases.slice(0,4))card.append(node('div',`Seed ${c.seed} · ${fmt(c.sim_time,0)}s · ${c.alive} alive${now-c.updated_at>90?' · stale':''}`,'run-detail'));$('fleet').append(card);}
  facts($('guardrails'),[['Elapsed / runtime limit',`${duration(b.elapsed_seconds)} / ${limits.max_hours}h`],['Last supervisor checkpoint',`${Math.floor(Math.max(0,now-data.checkpoint_at))}s ago`],['Shutdown watchdog',data.watchdog.observed?'Observed running':'Not observed running'],['Hard cutoff (your time)',date(data.watchdog.deadline)],['Campaign storage',`${fmt((data.storage.used_bytes||0)/2**30,2)} GiB / ${fmt((data.storage.limit_bytes||0)/2**30,0)} GiB active limit`],['Final holdout',data.holdout]]);
  if(data.orchestrator?.model){$('guardrails').append(node('dt','LLM configured for new reviews'),node('dd',`${data.orchestrator.model} · ${data.orchestrator.model_reasoning_effort}`),node('dt','LLM sign-in method'),node('dd','ChatGPT subscription'));}
  $('timeline').replaceChildren();
  for(let g=1;g<=data.schedule.major_generations;g++){
    const row=node('div',null,'generation-row');row.append(node('span',`G${g}`,'generation-label'));
    for(let sub=1;sub<=data.schedule.subgenerations;sub++){
      const name=`g${g}.${sub}`,round=data.rounds.find(r=>r.name===name);
      const past=g<s.major || (g===s.major && sub<s.sub) || (g===s.major && ['closing','major','final','done'].includes(s.stage));
      const current=g===s.major && sub===s.sub && s.stage==='sub' && s.status==='running';
      const cell=node('span',`${g}.${sub}`,'generation-cell'+(round?' done':current?' current':past?' skipped':''));
      cell.title=round?(round.improved?'Improvement promoted':'Incumbent retained'):current?'Current subgeneration':past?'Not completed; skipped or stopped':'Planned';row.append(cell);
    }
    const boundary=data.rounds.find(r=>r.name===`g${g}.major`);
    row.append(node('span','Lucas + BO','generation-cell boundary'+(boundary?' done':g===s.major && ['closing','major'].includes(s.stage)?' current':'')));
    $('timeline').append(row);
  }
  const pending=data.steps.filter(p=>['pending','running'].includes(p.status));$('steps').replaceChildren();
  if(!pending.length)empty($('steps'),s.status==='running'?'Supervisor is preparing the next step.':`Campaign ${s.status}.`);
  for(const step of pending){const e=node('div');e.append(node('span',step.name,'step-name'));const p=step.progress||{};e.append(node('span',(p.total!=null?`${p.completed}/${p.total} runs completed`:step.study?`${step.study.completed} candidates evaluated · ${step.study.method}`:'Working')+` · ${duration(step.deadline-now)} remaining for this step`,'step-detail'));$('steps').append(e);}
  $('active').replaceChildren();if(!data.active.length)empty($('active'),'No development simulation is currently reporting progress. The supervisor may be reviewing, comparing, or preparing a search.');
  for(const run of data.active){const stale=now-run.updated_at>90;const card=node('article',null,'run');const h=node('div',null,'run-head');h.append(node('strong',`Seed ${run.seed}`),node('span',`${run.engine || '—'}${stale?' · stale':''}`));const bar=node('div',null,'run-progress'),fill=node('div');fill.style.width=`${Math.min(100,(run.sim_time||0)/(run.horizon||3000)*100)}%`;bar.append(fill);card.append(h,bar,node('div',`${fmt(run.sim_time)} / ${fmt(run.horizon,0)} sim seconds`,'run-detail'),node('div',`${run.alive ?? '—'} alive · ${duration(run.wall_seconds)} wall time · ${run.id.slice(0,8)}`,'run-detail'));$('active').append(card);}
  options($('comparison'),data.comparisons.map(c=>[c.name,c.name]));renderComparison();
  const caseEntries=data.cases.map(c=>[c.id,`${c.engine || 'run'} · seed ${c.seed} · ${fmt(c.sim_time)}s · ${c.id.slice(0,8)}`]);
  options($('case'),caseEntries);const caseSignature=JSON.stringify(data.cases);if(caseSignature!==lastCases){lastCases=caseSignature;renderCase();}
  const reviews=JSON.stringify(data.reviews);if(reviews!==lastReviews){lastReviews=reviews;renderReviews();}
  const events=JSON.stringify(data.events);if(events!==lastEvents){lastEvents=events;$('events').replaceChildren();for(const e of [...data.events].reverse()){const row=node('div',null,'event');const t=node('time',e.time?new Date(e.time*1000).toLocaleTimeString(): '—');t.title=date(e.time);row.append(t,node('span',e.kind),node('span',e.name,'mono'));$('events').append(row);}if(!data.events.length)empty($('events'),'Research events will appear here.');}
  $('rounds').replaceChildren();for(const r of [...data.rounds].reverse().slice(0,5))$('rounds').append(node('div',`${r.name} · ${r.improved?'Promoted '+r.winner.label:'Retained '+r.before.label}${r.stop_early?' · early stop recommended':''}`,'round-result'));
}
function renderComparison(){
 const c=report?.comparisons.find(c=>c.name===$('comparison').value);$('comparison-chart').replaceChildren();$('comparison-table').replaceChildren();
 if(!c){set('comparison-note','Development comparisons will appear once the first batch completes.');return;}
 set('comparison-note',`${c.engine==='python'?'Python · authoritative development evaluation':'C++ fastsim · screening/control evidence'} · ${c.seeds.length} matched maps${c.control?' · equivalence check':''}. Bars show mean survival; comparisons across engines are not equivalent.`);
 const maximum=Math.max(1,...c.rows.map(r=>r.mean_survival || 0));
 for(const r of c.rows){const row=node('div',null,'chart-row'+(r.id===report.best.id?' best':''));const label=node('span',r.label,'chart-label');label.title=r.label;const track=node('div',null,'chart-track'),fill=node('div',null,'chart-fill');fill.style.width=`${(r.mean_survival||0)/maximum*100}%`;track.append(fill);row.append(label,track,node('span',`${fmt(r.mean_survival,0)}s`,'chart-value'));$('comparison-chart').append(row);}
 const table=node('table'),head=node('thead'),hr=node('tr');['Candidate','Mean survival','Worst survival','Mean score'].forEach(x=>hr.append(node('th',x)));head.append(hr);table.append(head);const body=node('tbody');for(const r of c.rows){const tr=node('tr');[r.label,fmt(r.mean_survival)+'s',fmt(r.worst_survival)+'s',fmt(r.mean_score)].forEach(v=>tr.append(node('td',v)));body.append(tr);}table.append(body);$('comparison-table').append(table);
}
function renderReviews(){
 $('reviews').replaceChildren();if(!report.reviews.length){empty($('reviews'),'The first LLM review is pending. Each completed review will add its evidence, hypotheses, compute/complexity assessment, and next direction here.');return;}
 for(const [i,r] of report.reviews.entries()){const details=node('details');details.open=i===0;details.append(node('summary',r.name),node('p',r.summary));for(const [key,title] of [['evidence','Evidence'],['hypotheses','Hypotheses to test'],['direction','Research direction'],['complexity','Complexity and compute'],['upstream','Lucas branch'],['focus_paths','Parameters to tune']]){const value=r[key];if(!value || (Array.isArray(value)&&!value.length))continue;details.append(node('h4',title));if(Array.isArray(value)){const list=node('ul');for(const v of value)list.append(node('li',typeof v==='string'?v:JSON.stringify(v)));details.append(list);}else details.append(node('p',typeof value==='string'?value:JSON.stringify(value)));}$('reviews').append(details);}
}
function renderCase(){
 const c=report?.cases.find(c=>c.id===$('case').value);$('diagnostics').replaceChildren();
 if(!c){empty($('diagnostics'),'Diagnostics appear as development runs finish.');$('screenshots').replaceChildren();return;}
 const d=c.diagnostics,t=d.totals||{};const grid=node('div',null,'diagnostic-grid');
 for(const [name,value,note] of [['Survival',fmt(c.sim_time)+'s',`${c.status || 'complete'} · ${c.final_agents ?? '—'} survivors`],['Energy gained / fruit',fmt(d.mean_absorbed_energy_per_fruit),`Gross ${fmt(d.mean_gross_energy_per_fruit)} before energy cap`],['Fruit eaten',fmt(t.fruits_eaten,0),`Energy lost to cap: ${fmt(t.fruit_cap_waste)}`],['Policy compute',fmt(c.policy_rpc_seconds_per_1000_agent_decisions,2)+'s','Wall time per 1,000 agent decisions']]){const card=node('div',null,'diagnostic-card');card.append(node('span',name),node('strong',value),node('span',note));grid.append(card);}
 const detail=node('div',null,'diagnostic-details');const deaths=node('div');deaths.append(node('h3','Recorded causes of death'));const dl=node('dl',null,'facts');facts(dl,Object.entries(d.deaths||{}).map(([k,v])=>[k.replaceAll('_',' '),fmt(v,0)]));if(!Object.keys(d.deaths||{}).length)dl.append(node('p','No death breakdown recorded.','muted'));deaths.append(dl);const energy=node('div');energy.append(node('h3','Energy ledger'));const el=node('dl',null,'facts');facts(el,[['Fruit absorbed',fmt(t.fruit_absorbed_energy)],['Movement',fmt(t.movement_energy)],['Turning',fmt(t.turning_energy)],['Maintenance',fmt(t.maintenance_energy)],['Reproduction',fmt(t.reproduction_energy)]]);energy.append(el);detail.append(deaths,energy);$('diagnostics').append(grid,detail);if(c.error)$('diagnostics').append(node('p',c.error,'notice'));
 const signature=c.id+JSON.stringify(d.screenshots||[]);if(signature!==selectedCase){selectedCase=signature;$('screenshots').replaceChildren();for(const file of (d.screenshots||[]).filter(f=>/^frame-\d+\.png$/.test(f)).slice(0,3)){const figure=node('figure'),img=node('img');img.alt=`Simulation screenshot for seed ${c.seed}, ${file}`;img.loading='lazy';img.src=`/api/image?case=${encodeURIComponent(c.id)}&file=${encodeURIComponent(file)}`;img.addEventListener('error',()=>{figure.replaceChildren(node('p','Screenshot unavailable while the Pod is disconnected.','muted'));});figure.append(img,node('figcaption',`${c.engine} · seed ${c.seed} · ${file}`));$('screenshots').append(figure);}}
}
async function refresh(){try{const response=await fetch('/api/state',{cache:'no-store'});if(!response.ok)throw new Error('Local dashboard is unavailable');const view=await response.json();const age=view.last_success?(Date.now()/1000-view.last_success):Infinity;const live=view.connected&&age<60;$('lamp').className='lamp'+(live?' live':'');set('connection',live?'Connected · auto-updating':view.data?'Offline · saved report':'Connecting');set('updated',view.last_success?`Last received ${date(view.last_success)} · refresh ${view.interval || 15}s`:'Waiting for the first report');const messages=[];if(view.error)messages.push(view.error);if(view.data){if(view.data.state.error)messages.push(`Campaign error: ${view.data.state.error}`);if(Date.now()/1000-view.data.checkpoint_at>90&&view.data.state.status==='running')messages.push('Supervisor checkpoint is stale; this may indicate a stalled or disconnected campaign.');render(view.data);}if(!live&&view.data)messages.push('This is the last successful snapshot, not live Pod status.');set('notice',messages.join(' '));}catch(error){set('connection','Local server disconnected');$('lamp').className='lamp';set('notice','The local dashboard server is unavailable. Previously displayed data is retained; restart research_dashboard.cmd to reconnect.');}finally{setTimeout(refresh,5000);}}
$('comparison').addEventListener('change',renderComparison);$('case').addEventListener('change',renderCase);refresh();
