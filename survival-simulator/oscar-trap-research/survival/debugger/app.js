/* Survival replay debugger. Local, dependency-free, and deliberately read-only. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const canvas = $('world');
  const ctx = canvas.getContext('2d');
  const chart = $('population-chart');
  const chartCtx = chart.getContext('2d');
  const TYPES = ['agents', 'predators', 'fruits', 'trees'];
  const SINGULAR = { agents: 'Agent', predators: 'Predator', fruits: 'Fruit', trees: 'Tree' };
  const PLURAL = { agent: 'agents', predator: 'predators', fruit: 'fruits', tree: 'trees' };
  const COLORS = { agents: '#b1f3bc', predators: '#ff8d76', fruits: '#f0cd73', trees: '#66957b' };
  const state = {
    replay: null, frameIndex: 0, time: 0, playing: false, speed: 1, selected: null,
    follow: false, camera: { x: 0, y: 0, scale: 1 }, fitScale: 1,
    background: null, width: 1, height: 1, dpr: 1, loadId: 0,
    frameMaps: new WeakMap(), lastRoster: '', lastEventKey: '', lastTick: null,
    dirty: true, pointer: null, eventTimes: [], population: [], manifest: [],
    nativeMode: false, nativeImages: new Map(), nativePending: new Set(),
    replayFile: '', replayName: '', manifestLoading: false, manifestReady: false, manifestSignature: '',
  };
  const portable = Array.isArray(window.SURVIVAL_BUNDLES);
  const options = { trails: true, vision: false, hearing: false, labels: false };
  let toastTimer;

  function finite(value, fallback = 0) { return Number.isFinite(value) ? value : fallback; }
  function number(value, digits = 1) {
    return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '–';
  }
  function clock(value, decimals = true) {
    value = Math.max(0, finite(value));
    const minutes = Math.floor(value / 60);
    const seconds = Math.floor(value % 60);
    const tenth = Math.floor((value % 1 + 0.00001) * 10);
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}${decimals ? `.${tenth}` : ''}`;
  }
  function entityKey(type, id) { return `${type}:${String(id)}`; }
  function currentFrame() { return state.replay?.frames[state.frameIndex]; }
  function frameMap(frame) {
    let map = state.frameMaps.get(frame);
    if (!map) {
      map = new Map();
      for (const type of TYPES) for (const entity of frame[type]) map.set(entityKey(type, entity.id), entity);
      state.frameMaps.set(frame, map);
    }
    return map;
  }
  function selectedEntity(frame = currentFrame()) {
    return frame && state.selected ? frameMap(frame).get(state.selected.key) : null;
  }
  function selectEntity(type, id) {
    state.selected = { type, id, key: entityKey(type, id) };
    state.lastEventKey = '';
    renderInspector();
    renderEvents();
    state.dirty = true;
  }
  function showError(message) {
    $('toast').textContent = message;
    $('toast').hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $('toast').hidden = true; }, 12000);
  }
  function setLoading(loading, message = 'Opening recording…') {
    $('loading-state').hidden = !loading;
    $('loading-text').textContent = message;
  }
  function setPlaying(playing) {
    state.playing = Boolean(playing && state.replay);
    $('play-pause').textContent = state.playing ? 'Ⅱ' : '▶';
    $('play-pause').setAttribute('aria-label', state.playing ? 'Pause' : 'Play');
    state.lastTick = null;
  }
  function togglePlayback() {
    if (!state.replay) return;
    if (!state.playing && state.time >= state.replay.frames.at(-1).t) seek(state.replay.frames[0].t);
    setPlaying(!state.playing);
  }

  // Gzip is decoded in the browser. Imported files never leave this page.
  async function decodeReplay(blob, name) {
    if (blob.size > 200 * 1024 * 1024) throw new Error('That file is over 200 MB. Record a shorter replay or a larger frame interval.');
    const magic = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
    let text;
    if (/\.gz$/i.test(name) || (magic[0] === 0x1f && magic[1] === 0x8b)) {
      if (!('DecompressionStream' in window)) throw new Error('This browser cannot open gzip files. Decompress the file first, then open the .json file.');
      const stream = blob.stream().pipeThrough(new DecompressionStream('gzip'));
      const reader = stream.getReader();
      const decoder = new TextDecoder();
      let bytes = 0;
      const parts = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        bytes += value.byteLength;
        if (bytes > 500 * 1024 * 1024) { await reader.cancel(); throw new Error('The expanded replay exceeds 500 MB. Use a shorter recording or larger frame interval.'); }
        parts.push(decoder.decode(value, { stream: true }));
      }
      parts.push(decoder.decode());
      text = parts.join('');
    } else text = await blob.text();
    let data;
    try { data = JSON.parse(text); } catch (_) { throw new Error('This file is not valid replay JSON. Open a file produced by the simulation recorder.'); }
    validateReplay(data);
    return data;
  }
  function validateReplay(data) {
    if (!data || data.format !== 'survival-replay' || data.version !== 1) throw new Error('Unsupported replay format. Expected survival-replay, version 1.');
    if (!data.world || !Number.isFinite(data.world.width) || !Number.isFinite(data.world.height) || data.world.width <= 0 || data.world.height <= 0) throw new Error('The replay is missing valid world dimensions.');
    if (!Array.isArray(data.frames) || !data.frames.length) throw new Error('This replay has no frames.');
    if (!data.meta || typeof data.meta !== 'object') data.meta = {};
    if (!data.summary || typeof data.summary !== 'object') data.summary = {};
    let previous = -Infinity;
    for (let i = 0; i < data.frames.length; i++) {
      const frame = data.frames[i];
      if (!frame || !Number.isFinite(frame.t) || frame.t < 0 || frame.t <= previous) throw new Error(`Frame ${i + 1} has an invalid or out-of-order time.`);
      previous = frame.t;
      for (const type of TYPES) {
        if (frame[type] == null) frame[type] = [];
        if (!Array.isArray(frame[type])) throw new Error(`Frame ${i + 1} has invalid ${type}.`);
        const ids = new Set();
        for (const entity of frame[type]) {
          if (!entity || !['string', 'number'].includes(typeof entity.id) || !Number.isFinite(entity.x) || !Number.isFinite(entity.y)) throw new Error(`Frame ${i + 1} contains a ${SINGULAR[type].toLowerCase()} without a stable ID or position.`);
          const id = String(entity.id);
          if (ids.has(id)) throw new Error(`Frame ${i + 1} contains duplicate ${type} IDs.`);
          ids.add(id);
        }
      }
    }
    data.events = (Array.isArray(data.events) ? data.events : []).filter(e => e && Number.isFinite(e.t)).sort((a, b) => a.t - b.t);
    data.world.obstacles = (Array.isArray(data.world.obstacles) ? data.world.obstacles : []).filter(o => o && Number.isFinite(o.x) && Number.isFinite(o.y) && Number.isFinite(o.width) && Number.isFinite(o.height));
  }
  async function loadBackground(src) {
    if (typeof src !== 'string' || !/^data:image\/(png|webp|jpeg);base64,/i.test(src)) return null;
    const image = new Image();
    image.src = src;
    try { await image.decode(); return image; } catch (_) { return null; }
  }
  async function openReplay(blob, name = 'Local replay', sampleFile = '', requestedLoadId = null) {
    const loadId = requestedLoadId ?? ++state.loadId;
    setPlaying(false);
    setLoading(true);
    $('toast').hidden = true;
    try {
      const replay = await decodeReplay(blob, name);
      const background = await loadBackground(replay.world.background);
      if (loadId !== state.loadId) return;
      state.replay = replay;
      state.replayFile = sampleFile;
      state.replayName = replay.meta.title || name.replace(/\.json(?:\.gz)?$/i, '');
      state.background = background;
      state.nativeImages = new Map();
      state.nativePending = new Set();
      state.nativeMode = replay.frames.every(frame => typeof frame.native_image === 'string' && /^data:image\/png;base64,/.test(frame.native_image));
      $('native-option').hidden = !state.nativeMode;
      $('show-native').checked = state.nativeMode;
      if (state.nativeMode) { options.trails = false; $('show-trails').checked = false; }
      updateRenderMode();
      state.frameIndex = 0;
      state.time = replay.frames[0].t;
      state.selected = null;
      state.follow = false;
      state.frameMaps = new WeakMap();
      state.lastRoster = '';
      state.lastEventKey = '';
      state.population = replay.frames.map(f => [f.t, f.agents.length]);
      state.eventTimes = replay.events.map(e => e.t);
      renderManifest(sampleFile);
      $('empty-state').hidden = true;
      $('play-pause').disabled = false;
      $('timeline').disabled = false;
      $('entity-select').disabled = false;
      $('timeline').min = String(replay.frames[0].t);
      $('timeline').max = String(replay.frames.at(-1).t || 1);
      $('duration').textContent = clock(replay.frames.at(-1).t);
      $('recording-title').textContent = replay.meta.title || name.replace(/\.json(?:\.gz)?$/i, '');
      const subtitle = [replay.meta.policy && `Policy: ${replay.meta.policy}`, replay.meta.seed != null && `Seed ${replay.meta.seed}`, replay.meta.record_interval != null && `${number(replay.meta.record_interval, 3)} s nominal capture`].filter(Boolean);
      $('recording-subtitle').textContent = subtitle.join(' · ') || 'Local simulation recording';
      renderMetadata();
      fitWorld();
      if (replay.meta.scenario === 'controlled') focusExperiment();
      updateUI(true);
      if (!background && replay.world.background) showError('The biome image could not be decoded. Entity positions and playback are still available.');
    } catch (error) {
      if (loadId === state.loadId) { renderManifest(state.replayFile); showError(error.message || 'Could not open the recording.'); }
    } finally { if (loadId === state.loadId) setLoading(false); }
  }
  async function loadSample(file) {
    if (!file) return;
    const loadId = ++state.loadId;
    setPlaying(false);
    setLoading(true, 'Loading sample recording…');
    try {
      const bundled = state.manifest.find(item => item.file === file && item.data);
      if (bundled) {
        const binary = atob(bundled.data);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        await openReplay(new Blob([bytes]), `${bundled.title}.json.gz`, file, loadId);
        return;
      }
      const url = new URL(file, document.baseURI);
      if (url.origin !== location.origin || (!url.pathname.startsWith(new URL('.', document.baseURI).pathname) && !url.pathname.startsWith('/replays/'))) throw new Error('Recordings must be served by this local replay server.');
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Could not load this sample (${response.status}). You can also use Open replay.`);
      const blob = await response.blob();
      if (loadId === state.loadId) await openReplay(blob, file.split('/').at(-1), file, loadId);
    } catch (error) { if (loadId === state.loadId) { setLoading(false); renderManifest(state.replayFile); showError(error.message); } }
  }
  function renderManifest(selectedFile = $('sample-select').value) {
    const select = $('sample-select');
    const fragment = document.createDocumentFragment();
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = state.replay && !state.replayFile ? `Opened file: ${state.replayName}` : 'Choose a recording…';
    fragment.append(placeholder);
    const titleCounts = new Map();
    for (const item of state.manifest) {
      const title = item.label || item.title || item.file;
      titleCounts.set(title, (titleCounts.get(title) || 0) + 1);
    }
    const groups = new Map();
    for (const item of state.manifest) {
      const option = document.createElement('option');
      const title = item.label || item.title || item.file;
      option.value = item.file;
      option.textContent = !item.label && titleCounts.get(title) > 1 ? `${title} · ${item.source || item.file}` : title;
      option.title = [item.description, item.source].filter(Boolean).join(' · ');
      if (item.group) {
        if (!groups.has(item.group)) {
          const group = document.createElement('optgroup');
          group.label = String(item.group);
          groups.set(item.group, group);
          fragment.append(group);
        }
        groups.get(item.group).append(option);
      } else fragment.append(option);
    }
    if (state.replayFile && !state.manifest.some(item => item.file === state.replayFile)) {
      const retained = document.createElement('option');
      retained.value = state.replayFile;
      retained.textContent = `${state.replayName} · loaded, no longer in list`;
      retained.disabled = true;
      fragment.append(retained);
    }
    select.replaceChildren(fragment);
    const wanted = [...select.options].some(option => option.value === selectedFile) ? selectedFile : state.replayFile;
    select.value = wanted || '';
  }
  function catalogStatus(message, detail = '') {
    $('catalog-status').textContent = message;
    $('catalog-status').title = detail;
  }
  async function loadManifest(initial = false) {
    if (state.manifestLoading) return;
    state.manifestLoading = true;
    $('refresh-recordings').disabled = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      let entries;
      if (portable) {
        entries = window.SURVIVAL_BUNDLES.filter(item => item && typeof item.data === 'string').map((item, index) => ({ ...item, file: `bundle:${index}` }));
      } else {
        const response = await fetch('recordings/manifest.json', { cache: 'no-store', signal: controller.signal });
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        const data = await response.json();
        if (!Array.isArray(data.recordings)) throw new Error('Invalid recording list');
        if (data.stats?.scanning) {
          catalogStatus('Scanning recordings…', 'The server is discovering completed runs. The current replay and list stay unchanged.');
          return;
        }
        entries = data.recordings;
      }
      const seen = new Set();
      state.manifest = entries.filter(item => {
        if (!item || typeof item.file !== 'string' || !item.file || seen.has(item.file)) return false;
        seen.add(item.file);
        return true;
      });
      state.manifestReady = true;
      const signature = JSON.stringify(state.manifest.map(item => [item.file, item.label, item.title, item.description, item.source, item.group]));
      if (signature !== state.manifestSignature) {
        state.manifestSignature = signature;
        renderManifest();
      }
      const count = `${state.manifest.length} recording${state.manifest.length === 1 ? '' : 's'}`;
      catalogStatus(`${count} · ${portable ? 'bundled' : 'live'}`, portable ? 'This portable copy contains its bundled recordings.' : 'Checks for completed recordings every 10 seconds while visible.');
      if (initial && state.manifest.length && !state.replay && !state.loadId) await loadSample(state.manifest[0].file);
    } catch (error) {
      const detail = error.name === 'AbortError' ? 'The replay server took too long to respond.' : error.message;
      catalogStatus(state.manifestReady ? `${state.manifest.length} recordings · refresh failed` : 'Recording list unavailable', `${detail} The current replay is unchanged. Use Refresh list to retry.`);
      if (!state.replay && !state.manifestReady) {
        $('sample-select').options[0].textContent = 'Open a replay file to begin';
        $('recording-subtitle').textContent = location.protocol === 'file:' ? 'Use Open replay, or run the local server to browse recordings.' : 'Open a JSON or compressed JSON replay, or drop it anywhere.';
      }
    } finally {
      clearTimeout(timeout);
      state.manifestLoading = false;
      $('refresh-recordings').disabled = false;
    }
  }

  function frameIndexAt(time) {
    const frames = state.replay.frames;
    let low = 0, high = frames.length - 1;
    while (low < high) {
      const mid = Math.ceil((low + high) / 2);
      if (frames[mid].t <= time + 1e-8) low = mid; else high = mid - 1;
    }
    return low;
  }
  function seek(time) {
    if (!state.replay) return;
    state.time = Math.max(state.replay.frames[0].t, Math.min(state.replay.frames.at(-1).t, finite(time)));
    const index = frameIndexAt(state.time);
    const changed = index !== state.frameIndex;
    state.frameIndex = index;
    updateUI(changed);
    state.dirty = true;
  }
  function step(delta) {
    if (!state.replay) return;
    setPlaying(false);
    const index = Math.max(0, Math.min(state.replay.frames.length - 1, state.frameIndex + delta));
    seek(state.replay.frames[index].t);
  }
  function updateUI(frameChanged = false) {
    const frame = currentFrame();
    if (!frame) return;
    $('timeline').value = String(state.time);
    $('current-time').textContent = clock(state.time);
    $('sim-time').textContent = clock(frame.t);
    $('frame-label').textContent = `Frame ${state.frameIndex + 1} / ${state.replay.frames.length}`;
    if (frameChanged) {
      $('score').textContent = number(frame.score, 0);
      $('agent-count').textContent = String(frame.agents.length);
      $('predator-count').textContent = String(frame.predators.length);
      $('fruit-count').textContent = String(frame.fruits.length);
      $('population-label').textContent = `${frame.agents.length} alive`;
      renderRoster();
      renderInspector();
      renderEvents();
    }
    drawPopulation();
  }
  function renderRoster() {
    const frame = currentFrame();
    const rosterKey = [...frame.agents.map(e => `a${e.id}`), ...frame.predators.map(e => `p${e.id}`)].join('|');
    if (rosterKey === state.lastRoster) return;
    state.lastRoster = rosterKey;
    const select = $('entity-select');
    select.replaceChildren();
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'Select an agent or predator…';
    select.append(placeholder);
    for (const type of ['agents', 'predators']) {
      const group = document.createElement('optgroup');
      group.label = type === 'agents' ? 'Agents alive' : 'Predators';
      for (const entity of frame[type]) {
        const option = document.createElement('option');
        option.value = entityKey(type, entity.id);
        option.textContent = `${SINGULAR[type]} ${entity.id}`;
        group.append(option);
      }
      select.append(group);
    }
    select.value = state.selected?.key || '';
  }
  function addStat(parent, label, value) {
    const row = document.createElement('div');
    row.className = 'stat';
    const dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = label;
    dd.textContent = String(value ?? '–');
    row.append(dt, dd);
    parent.append(row);
  }
  function renderInspector() {
    const selected = state.selected;
    $('inspector-empty').hidden = Boolean(selected);
    $('entity-details').hidden = !selected;
    if (!selected) {
      $('perception-note').textContent = 'Select an agent or predator for sensing overlays';
      return;
    }
    let entity = selectedEntity();
    const present = Boolean(entity);
    let seenTime = currentFrame().t;
    if (!entity) {
      for (let i = state.frameIndex - 1; i >= 0; i--) {
        entity = frameMap(state.replay.frames[i]).get(selected.key);
        if (entity) { seenTime = state.replay.frames[i].t; break; }
      }
    }
    $('entity-select').value = present ? selected.key : '';
    const isCreature = ['agents', 'predators'].includes(selected.type);
    $('entity-type').textContent = SINGULAR[selected.type].toUpperCase();
    $('entity-type').style.color = COLORS[selected.type];
    $('entity-name').textContent = `${SINGULAR[selected.type]} ${selected.id}`;
    $('follow').disabled = !present;
    $('follow').setAttribute('aria-pressed', String(state.follow));
    $('follow').textContent = state.follow ? 'Following' : 'Follow';
    $('entity-status').textContent = present ? `${isCreature ? (entity.resting ? 'Resting' : 'Active') : 'Present'}${entity.biome != null ? ` · ${entity.biome}` : ''}` : entity ? `Absent from this frame · last seen ${clock(seenTime)}` : 'Not yet present in this recording';
    $('stats').replaceChildren();
    $('energy-block').hidden = !entity || !Number.isFinite(entity.energy);
    $('decision-block').hidden = selected.type !== 'agents' || !entity;
    $('action-block').hidden = selected.type !== 'agents' || !entity;
    $('observations-block').hidden = selected.type !== 'agents' || !entity;
    if (!entity) return;
    if (Number.isFinite(entity.energy)) {
      const maximum = Number.isFinite(entity.max_energy) ? entity.max_energy : 60;
      $('energy-label').textContent = `${number(entity.energy)}${isCreature ? ` / ${number(maximum)}` : ' units'}`;
      const ratio = Math.max(0, Math.min(1, entity.energy / Math.max(1, maximum)));
      $('energy-fill').style.width = `${ratio * 100}%`;
      $('energy-fill').style.background = ratio < 0.15 ? '#ff8d76' : COLORS[selected.type];
    }
    const stats = $('stats');
    addStat(stats, 'Position · x, y', `${number(entity.x)}, ${number(entity.y)}`);
    if (Number.isFinite(entity.age)) addStat(stats, 'Age', `${number(entity.age)} s${Number.isFinite(entity.max_age) ? ` / ${number(entity.max_age)} s` : ''}`);
    if (isCreature) {
      addStat(stats, 'Walking speed', `${number(entity.speed, 2)} units/tick`);
      addStat(stats, 'Sprint speed', `${number(entity.sprint_speed, 2)} units/tick`);
      addStat(stats, 'Hearing radius', `${number(entity.hearing_radius)} units`);
      addStat(stats, 'Vision range', `${number(entity.vision_range)} units`);
      addStat(stats, 'Vision angle', `${number(finite(entity.vision_angle) * 180 / Math.PI)}°`);
      addStat(stats, 'Heading', `${number(((finite(entity.direction) * 180 / Math.PI) % 360 + 360) % 360)}°`);
    }
    if (Number.isFinite(entity.size)) addStat(stats, 'Size', `${number(entity.size)} units`);
    if (Number.isFinite(entity.radius)) addStat(stats, 'Radius', `${number(entity.radius)} units`);
    if (selected.type === 'fruits') addStat(stats, 'Maturity', entity.energy >= 60 ? 'Fully grown' : 'Still growing');
    if (selected.type === 'agents') {
      const decision = entity.decision;
      $('decision-rule').textContent = decision?.rule || 'No decision annotation';
      const timing = Number.isFinite(entity.action_t) ? `Action at ${clock(entity.action_t)}. ` : '';
      $('decision-detail').textContent = timing + (decision?.detail || 'Load a replay recorded with policy annotations to see its rule and reason.');
      $('action-json').textContent = entity.action ? JSON.stringify(entity.action, null, 2) : 'No preceding action recorded.';
      const observations = entity.action_observations ?? entity.observations;
      $('observation-count').textContent = Array.isArray(observations) ? `${observations.length} objects` : '';
      $('observations-json').textContent = observations != null ? JSON.stringify(observations, null, 2) : 'No observations recorded.';
      $('observations-block').querySelector('.data-note').textContent = entity.action_observations != null ? `Exact recorded inputs to the last action${Number.isFinite(entity.action_t) ? ` at ${clock(entity.action_t)}` : ''}. The world above shows the state after that action.` : 'Engine observation objects cached at this frame. These may precede the displayed world state.';
    }
    $('state-note').textContent = selected.type === 'agents' ? 'Positions and stats above are debug state. They are not all available to the policy. Decision annotations describe policy rules, not hidden reasoning.' : `${selected.type === 'trees' || selected.type === 'predators' ? 'This stable ID is assigned by the recorder. ' : ''}Positions and stats are omniscient debug state.`;
    $('perception-note').textContent = isCreature ? 'Selected entity · nominal ranges; vision may be occluded by walls' : 'Select an agent or predator for sensing overlays';
  }
  function eventMatchesSelected(event) {
    return state.selected && PLURAL[event.entity_type] === state.selected.type && String(event.entity_id) === String(state.selected.id);
  }
  function renderEvents() {
    if (!state.replay) return;
    let events = state.replay.events;
    const selectedOnly = $('selected-events').checked;
    if (selectedOnly) events = events.filter(eventMatchesSelected);
    const eventTime = currentFrame().t;
    let split = 0;
    while (split < events.length && events[split].t <= eventTime + 1e-8) split++;
    const start = Math.max(0, split - 6);
    const visible = events.slice(start, start + 35);
    const key = `${selectedOnly}:${state.selected?.key}:${start}:${split}:${events.length}`;
    if (key === state.lastEventKey) return;
    state.lastEventKey = key;
    const container = $('events');
    container.replaceChildren();
    if (!events.length) {
      const empty = document.createElement('p');
      empty.className = 'muted';
      empty.textContent = selectedOnly ? (state.selected ? 'No events recorded for this entity.' : 'Select an entity to filter events.') : 'This recording has no event annotations.';
      container.append(empty);
      return;
    }
    if (start > 0) {
      const note = document.createElement('p');
      note.className = 'event-overflow';
      note.textContent = `${start} earlier events · scrub back to explore`;
      container.append(note);
    }
    for (const event of visible) {
      const button = document.createElement('button');
      button.className = `event${event.t > eventTime + 1e-8 ? ' future' : ''}${event === events[split - 1] ? ' current' : ''}`;
      const time = document.createElement('span'), dot = document.createElement('span'), description = document.createElement('span');
      time.className = 'event-time';
      time.textContent = clock(event.t, false);
      dot.className = 'event-dot';
      if (['birth', 'death', 'fruit_eaten'].includes(event.type)) dot.classList.add(event.type);
      description.className = 'event-description';
      description.textContent = event.text || `${event.type || 'Event'}${event.entity_id != null ? ` · ${event.entity_type} ${event.entity_id}` : ''}`;
      button.append(time, dot, description);
      button.title = `${clock(event.t)} · ${description.textContent}`;
      button.addEventListener('click', () => {
        setPlaying(false);
        // Seek to the first recorded frame at or after an event, so births are visible.
        let index = frameIndexAt(event.t);
        if (state.replay.frames[index].t + 1e-8 < event.t) index = Math.min(index + 1, state.replay.frames.length - 1);
        seek(state.replay.frames[index].t);
        if (event.entity_id != null && PLURAL[event.entity_type]) selectEntity(PLURAL[event.entity_type], event.entity_id);
        if (Number.isFinite(event.x) && Number.isFinite(event.y) && !state.follow) {
          state.camera.x = event.x;
          state.camera.y = event.y;
          state.dirty = true;
        }
      });
      container.append(button);
    }
    if (start + visible.length < events.length) {
      const note = document.createElement('p');
      note.className = 'event-overflow';
      note.textContent = `${events.length - start - visible.length} later events`;
      container.append(note);
    }
  }
  function renderMetadata() {
    const { meta, world, summary, frames } = state.replay;
    const rows = [
      ['Policy', meta.policy], ['Seed', meta.seed], ['Scenario', meta.scenario],
      ['World', `${world.width} × ${world.height} units`], ['Simulation tick', meta.dt != null ? `${meta.dt} s` : null],
      ['Nominal capture interval', meta.record_interval != null ? `${meta.record_interval} s + event frames` : null],
      ['Frames', frames.length], ['Source commit', meta.source_commit], ['Policy SHA-256', meta.policy_sha256],
      ['Created', meta.created_at], ['Platform', meta.platform], ['Stop reason', summary.reason], ['Notes', meta.notes],
      ['Rendering', meta.renderer || 'Lightweight Canvas debug view'], ['Native resolution', meta.native_resolution?.join(' × ')], ['Rendering notes', meta.rendering_note],
    ];
    const container = $('metadata');
    container.replaceChildren();
    for (const [label, value] of rows) {
      if (value == null) continue;
      const row = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
      dt.textContent = label;
      dd.textContent = typeof value === 'object' ? JSON.stringify(value) : String(value);
      row.append(dt, dd);
      container.append(row);
    }
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    state.width = Math.max(1, rect.width);
    state.height = Math.max(1, rect.height);
    state.dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(state.width * state.dpr);
    canvas.height = Math.round(state.height * state.dpr);
    const chartRect = chart.getBoundingClientRect();
    chart.width = Math.round(chartRect.width * state.dpr);
    chart.height = Math.round(chartRect.height * state.dpr);
    if (state.replay) {
      const oldFit = state.fitScale;
      state.fitScale = Math.min((state.width - 50) / state.replay.world.width, (state.height - 60) / state.replay.world.height);
      state.fitScale = Math.max(0.0001, state.fitScale);
      if (oldFit) state.camera.scale *= state.fitScale / oldFit;
      drawPopulation();
    }
    state.dirty = true;
  }
  function fitWorld() {
    if (!state.replay) return;
    const { width, height } = state.replay.world;
    state.fitScale = Math.max(0.0001, Math.min((state.width - 50) / width, (state.height - 60) / height));
    state.camera = { x: width / 2, y: height / 2, scale: state.fitScale };
    state.follow = false;
    renderInspector();
    state.dirty = true;
  }
  function focusExperiment() {
    const frame = currentFrame();
    if (!frame) return;
    const points = [...frame.agents, ...frame.predators];
    for (const obstacle of state.replay.world.obstacles) {
      points.push({ x: obstacle.x, y: obstacle.y });
      points.push({ x: obstacle.x + obstacle.width, y: obstacle.y + obstacle.height });
    }
    if (!points.length) return;
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
    state.camera.x = (left + right) / 2;
    state.camera.y = (top + bottom) / 2;
    state.camera.scale = Math.max(state.fitScale, Math.min((state.width - 70) / (right - left + 180), (state.height - 70) / (bottom - top + 180)));
    state.dirty = true;
  }
  function worldPoint(x, y) {
    return { x: (x - state.width / 2) / state.camera.scale + state.camera.x, y: (y - state.height / 2) / state.camera.scale + state.camera.y };
  }
  function zoom(factor, x = state.width / 2, y = state.height / 2) {
    if (!state.replay) return;
    const anchor = worldPoint(x, y);
    state.camera.scale = Math.max(state.fitScale * 0.6, Math.min(state.fitScale * 40, state.camera.scale * factor));
    const after = worldPoint(x, y);
    state.camera.x += anchor.x - after.x;
    state.camera.y += anchor.y - after.y;
    state.dirty = true;
  }
  function drawWorld() {
    state.dirty = false;
    const { dpr, width, height, camera } = state;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0e1617';
    ctx.fillRect(0, 0, width, height);
    const frame = currentFrame();
    if (!frame) return;
    const selected = selectedEntity(frame);
    if (state.follow && selected) { camera.x = selected.x; camera.y = selected.y; }
    $('zoom-label').textContent = `${Math.round(camera.scale / state.fitScale * 100)}%`;
    ctx.translate(width / 2, height / 2);
    ctx.scale(camera.scale, camera.scale);
    ctx.translate(-camera.x, -camera.y);
    const world = state.replay.world;
    ctx.fillStyle = '#273f31';
    ctx.fillRect(0, 0, world.width, world.height);
    if (state.nativeMode) {
      const native = nativeImage(state.frameIndex);
      if (native) ctx.drawImage(native, 0, 0, world.width, world.height);
      nativeImage(state.frameIndex + 1);
      $('stage-hint').textContent = native ? 'Original simulator pixels · click to inspect · scroll to zoom' : 'Decoding simulator frame…';
    } else if (state.background) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(state.background, 0, 0, world.width, world.height);
      ctx.fillStyle = '#071b173d';
      ctx.fillRect(0, 0, world.width, world.height);
    } else drawGrid(world);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, world.width, world.height);
    ctx.clip();
    for (const obstacle of state.nativeMode ? [] : world.obstacles) {
      ctx.fillStyle = '#172322';
      ctx.fillRect(obstacle.x + 2, obstacle.y + 3, obstacle.width, obstacle.height);
      ctx.fillStyle = '#9ca8a0';
      ctx.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
      ctx.strokeStyle = '#dce2d177';
      ctx.lineWidth = 1 / camera.scale;
      ctx.strokeRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
    }
    if (options.trails) drawTrails(frame);
    if (!state.nativeMode) {
      if (selected && ['agents', 'predators'].includes(state.selected.type)) drawSenses(selected, state.selected.type);
      for (const entity of frame.trees) drawEntity(entity, 'trees');
      for (const entity of frame.fruits) drawEntity(entity, 'fruits');
      for (const entity of frame.predators) drawEntity(entity, 'predators');
      for (const entity of frame.agents) drawEntity(entity, 'agents');
    } else if (options.labels) {
      for (const type of ['agents', 'predators']) for (const entity of frame[type]) {
        if (state.selected?.key !== entityKey(type, entity.id)) drawLabel(entity, `${SINGULAR[type]} ${entity.id}`);
      }
    }
    if (selected) {
      const radius = Math.max(finite(selected.size, finite(selected.radius, 4)), 3 / camera.scale) + 6 / camera.scale;
      ctx.strokeStyle = '#f1fff0';
      ctx.lineWidth = 1.5 / camera.scale;
      ctx.beginPath();
      ctx.arc(selected.x, selected.y, radius, 0, Math.PI * 2);
      ctx.stroke();
      drawLabel(selected, `${SINGULAR[state.selected.type]} ${selected.id}`, true);
    }
    ctx.restore();
    ctx.strokeStyle = '#89a49466';
    ctx.lineWidth = 1 / camera.scale;
    ctx.strokeRect(0, 0, world.width, world.height);
    drawScale();
  }
  function updateRenderMode() {
    $('show-vision').disabled = state.nativeMode;
    $('show-hearing').disabled = state.nativeMode;
    $('show-vision').title = $('show-hearing').title = state.nativeMode ? 'The simulator already draws its own hearing and vision overlays.' : '';
    document.querySelector('.stage-legend').hidden = state.nativeMode;
    $('native-renderer-note').hidden = !state.nativeMode;
    $('stage-hint').textContent = state.nativeMode ? 'Original simulator pixels · click to inspect · scroll to zoom' : 'Click to inspect · drag to pan · scroll to zoom';
    state.dirty = true;
  }
  function nativeImage(index) {
    if (state.nativeImages.has(index)) return state.nativeImages.get(index);
    if (state.nativePending.has(index) || !state.replay.frames[index]) return null;
    const replay = state.replay;
    state.nativePending.add(index);
    loadBackground(replay.frames[index].native_image).then(image => {
      if (state.replay !== replay) return;
      state.nativePending.delete(index);
      if (image) {
        state.nativeImages.set(index, image);
        while (state.nativeImages.size > 12) state.nativeImages.delete(state.nativeImages.keys().next().value);
        state.dirty = true;
      } else {
        state.nativeMode = false;
        $('show-native').checked = false;
        updateRenderMode();
        showError('A simulator image could not be decoded. Showing the lightweight debug view.');
      }
    });
    return null;
  }
  function drawGrid(world) {
    ctx.strokeStyle = '#46624933';
    ctx.lineWidth = 1 / state.camera.scale;
    ctx.beginPath();
    for (let x = 0; x < world.width; x += 50) { ctx.moveTo(x, 0); ctx.lineTo(x, world.height); }
    for (let y = 0; y < world.height; y += 50) { ctx.moveTo(0, y); ctx.lineTo(world.width, y); }
    ctx.stroke();
  }
  function drawTrails(frame) {
    const frames = state.replay.frames;
    const fromTime = frame.t - 12;
    const from = Math.max(0, frameIndexAt(Math.max(frames[0].t, fromTime)));
    const stride = Math.max(1, Math.floor((state.frameIndex - from) / 60));
    const types = state.selected && ['agents', 'predators'].includes(state.selected.type) ? [state.selected.type] : ['agents', 'predators'];
    for (const type of types) {
      const entities = state.selected ? frame[type].filter(e => String(e.id) === String(state.selected.id)) : frame[type];
      ctx.strokeStyle = COLORS[type] + (state.selected ? 'b3' : '5c');
      ctx.lineWidth = (state.selected ? 1.6 : 1) / state.camera.scale;
      for (const entity of entities) {
        const key = entityKey(type, entity.id);
        ctx.beginPath();
        let started = false;
        for (let i = from; i <= state.frameIndex; i += stride) {
          const historic = frameMap(frames[i]).get(key);
          if (!historic) { started = false; continue; }
          if (!started) { ctx.moveTo(historic.x, historic.y); started = true; } else ctx.lineTo(historic.x, historic.y);
        }
        ctx.lineTo(entity.x, entity.y);
        ctx.stroke();
      }
    }
  }
  function drawSenses(entity, type) {
    const scale = state.camera.scale;
    if (options.vision && finite(entity.vision_range) > 0) {
      const angle = Math.max(0, Math.min(Math.PI * 2, finite(entity.vision_angle)));
      ctx.beginPath();
      ctx.moveTo(entity.x, entity.y);
      ctx.arc(entity.x, entity.y, entity.vision_range, finite(entity.direction) - angle / 2, finite(entity.direction) + angle / 2);
      ctx.closePath();
      ctx.fillStyle = COLORS[type] + '17';
      ctx.fill();
      ctx.strokeStyle = COLORS[type] + '77';
      ctx.lineWidth = 1 / scale;
      ctx.stroke();
    }
    if (options.hearing && finite(entity.hearing_radius) > 0) {
      ctx.beginPath();
      ctx.arc(entity.x, entity.y, entity.hearing_radius, 0, Math.PI * 2);
      ctx.fillStyle = '#9ccbdd0d';
      ctx.fill();
      ctx.strokeStyle = '#b2e8f399';
      ctx.lineWidth = 1 / scale;
      ctx.setLineDash([4 / scale, 4 / scale]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  function drawEntity(entity, type) {
    const scale = state.camera.scale;
    const radius = Math.max(finite(entity.size, finite(entity.radius, 4)), (type === 'fruits' ? 2 : 3) / scale);
    if (type === 'trees') {
      ctx.beginPath();
      ctx.arc(entity.x, entity.y, radius + 3, 0, Math.PI * 2);
      ctx.fillStyle = '#37644688';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(entity.x, entity.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = '#356247';
      ctx.fill();
      ctx.strokeStyle = '#9cbd765c';
      ctx.lineWidth = 1 / scale;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(entity.x, entity.y, Math.max(1.5, radius * 0.22), 0, Math.PI * 2);
      ctx.fillStyle = '#b09062';
      ctx.fill();
    } else if (type === 'fruits') {
      ctx.beginPath();
      ctx.arc(entity.x, entity.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = entity.energy >= 60 ? '#f0cd73' : '#bed580';
      ctx.fill();
      ctx.strokeStyle = '#293522cc';
      ctx.lineWidth = 0.7 / scale;
      ctx.stroke();
    } else {
      ctx.save();
      ctx.translate(entity.x, entity.y);
      ctx.rotate(finite(entity.direction));
      ctx.beginPath();
      if (type === 'predators') {
        ctx.moveTo(radius * 1.35, 0);
        ctx.lineTo(-radius * 0.65, radius);
        ctx.lineTo(-radius * 0.35, 0);
        ctx.lineTo(-radius * 0.65, -radius);
        ctx.closePath();
      } else ctx.arc(0, 0, radius, 0, Math.PI * 2);
      ctx.fillStyle = entity.resting ? '#b88272' : COLORS[type];
      ctx.fill();
      ctx.strokeStyle = '#163021';
      ctx.lineWidth = 1 / scale;
      ctx.stroke();
      if (type === 'agents') {
        ctx.beginPath();
        ctx.moveTo(radius * 0.3, -radius * 0.45);
        ctx.lineTo(radius * 0.86, 0);
        ctx.lineTo(radius * 0.3, radius * 0.45);
        ctx.strokeStyle = '#204c30';
        ctx.lineWidth = Math.max(radius * 0.16, 0.65 / scale);
        ctx.stroke();
      }
      ctx.restore();
      if (entity.resting && scale > state.fitScale * 2) drawLabel(entity, 'rest');
    }
    if (options.labels && ['agents', 'predators'].includes(type) && state.selected?.key !== entityKey(type, entity.id)) drawLabel(entity, String(entity.id));
  }
  function drawLabel(entity, label, selected = false) {
    const scale = state.camera.scale;
    const radius = finite(entity.size, finite(entity.radius, 4));
    ctx.save();
    ctx.translate(entity.x, entity.y - radius - 8 / scale);
    ctx.scale(1 / scale, 1 / scale);
    ctx.font = `${selected ? '500 ' : ''}10px -apple-system, sans-serif`;
    const w = ctx.measureText(label).width + 12;
    ctx.fillStyle = selected ? '#18251af0' : '#142218b3';
    ctx.fillRect(-w / 2, -13, w, 17);
    ctx.fillStyle = selected ? '#f0fff0' : '#cae4cd';
    ctx.textAlign = 'center';
    ctx.fillText(label, 0, -1);
    ctx.restore();
  }
  function drawScale() {
    ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    const target = 65 / state.camera.scale;
    const magnitude = 10 ** Math.floor(Math.log10(target));
    const units = [1, 2, 5, 10].map(n => n * magnitude).reduce((best, n) => Math.abs(n - target) < Math.abs(best - target) ? n : best, magnitude);
    const length = units * state.camera.scale;
    const x = 18, y = state.height - 22;
    ctx.fillStyle = '#112018b8';
    ctx.fillRect(x - 6, y - 20, length + 12, 30);
    ctx.strokeStyle = '#afc9b1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y - 3); ctx.lineTo(x, y + 2); ctx.lineTo(x + length, y + 2); ctx.lineTo(x + length, y - 3);
    ctx.stroke();
    ctx.fillStyle = '#b8cdbd';
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${number(units)} units`, x + length / 2, y - 7);
  }
  function drawPopulation() {
    if (!state.replay) return;
    const w = chart.width / state.dpr, h = chart.height / state.dpr;
    chartCtx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    chartCtx.clearRect(0, 0, w, h);
    const points = state.population;
    const first = points[0][0], duration = Math.max(0.001, points.at(-1)[0] - first);
    let max = 1;
    for (const point of points) max = Math.max(max, point[1]);
    chartCtx.beginPath();
    chartCtx.moveTo(0, h);
    const stride = Math.max(1, Math.floor(points.length / Math.max(1, w * 2)));
    for (let i = 0; i < points.length; i += stride) chartCtx.lineTo((points[i][0] - first) / duration * w, h - points[i][1] / max * (h - 3));
    chartCtx.lineTo(w, h - points.at(-1)[1] / max * (h - 3));
    chartCtx.lineTo(w, h);
    chartCtx.closePath();
    chartCtx.fillStyle = '#8dc69e24';
    chartCtx.fill();
    chartCtx.strokeStyle = '#8dc69e77';
    chartCtx.lineWidth = 1;
    chartCtx.stroke();
    const x = (state.time - first) / duration * w;
    chartCtx.fillStyle = '#b1f3bcbb';
    chartCtx.fillRect(x, 0, 1, h);
  }
  function pickEntity(x, y) {
    const frame = currentFrame();
    if (!frame) return;
    const point = worldPoint(x, y);
    let match = null, best = Infinity;
    for (const type of TYPES) {
      for (const entity of frame[type]) {
        const distance = Math.hypot(entity.x - point.x, entity.y - point.y);
        const radius = Math.max(finite(entity.size, finite(entity.radius, 4)), 3 / state.camera.scale);
        const score = Math.max(0, distance - radius) * state.camera.scale + (['agents', 'predators'].includes(type) ? 0 : 3);
        if (distance <= radius + 8 / state.camera.scale && score < best) { best = score; match = { type, entity }; }
      }
    }
    if (match) selectEntity(match.type, match.entity.id);
  }

  $('open-file').addEventListener('click', () => $('file-input').click());
  $('empty-open').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', event => {
    const file = event.target.files[0];
    if (file) openReplay(file, file.name);
    event.target.value = '';
  });
  $('sample-select').addEventListener('change', event => loadSample(event.target.value));
  $('refresh-recordings').addEventListener('click', () => loadManifest());
  $('play-pause').addEventListener('click', togglePlayback);
  $('restart').addEventListener('click', () => { if (state.replay) { setPlaying(false); seek(state.replay.frames[0].t); } });
  $('step-back').addEventListener('click', () => step(-1));
  $('step-forward').addEventListener('click', () => step(1));
  $('timeline').addEventListener('input', event => { setPlaying(false); seek(Number(event.target.value)); });
  $('speed').addEventListener('change', event => { state.speed = Number(event.target.value); });
  $('fit-world').addEventListener('click', fitWorld);
  $('zoom-in').addEventListener('click', () => zoom(1.35));
  $('zoom-out').addEventListener('click', () => zoom(1 / 1.35));
  $('follow').addEventListener('click', () => { state.follow = !state.follow; renderInspector(); state.dirty = true; });
  $('entity-select').addEventListener('change', event => {
    const key = event.target.value;
    if (!key) return;
    const type = key.split(':')[0];
    const entity = currentFrame()[type].find(e => entityKey(type, e.id) === key);
    if (entity) selectEntity(type, entity.id);
  });
  $('selected-events').addEventListener('change', () => { state.lastEventKey = ''; renderEvents(); });
  $('show-native').addEventListener('change', event => { state.nativeMode = event.target.checked; updateRenderMode(); });
  for (const option of ['trails', 'vision', 'hearing', 'labels']) {
    $('show-' + option).addEventListener('change', event => { options[option] = event.target.checked; state.dirty = true; });
  }
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    zoom(Math.exp(-Math.max(-100, Math.min(100, event.deltaY)) * 0.005), event.clientX - rect.left, event.clientY - rect.top);
  }, { passive: false });
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    state.pointer = { id: event.pointerId, x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, localX: event.clientX - rect.left, localY: event.clientY - rect.top, moved: false };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    const pointer = state.pointer;
    if (!pointer || pointer.id !== event.pointerId || !state.replay) return;
    const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
    if (Math.hypot(event.clientX - pointer.startX, event.clientY - pointer.startY) > 4) pointer.moved = true;
    if (pointer.moved) {
      state.camera.x -= dx / state.camera.scale;
      state.camera.y -= dy / state.camera.scale;
      if (state.follow) { state.follow = false; renderInspector(); }
      canvas.style.cursor = 'grabbing';
      state.dirty = true;
    }
    pointer.x = event.clientX;
    pointer.y = event.clientY;
  });
  canvas.addEventListener('pointerup', event => {
    if (state.pointer?.id !== event.pointerId) return;
    if (!state.pointer.moved) pickEntity(state.pointer.localX, state.pointer.localY);
    state.pointer = null;
    canvas.style.cursor = 'crosshair';
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointercancel', () => { state.pointer = null; canvas.style.cursor = 'crosshair'; });
  document.addEventListener('keydown', event => {
    if (event.target.matches('input,select,textarea') || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.code === 'Space') {
      if (event.target.matches('button,summary')) return;
      event.preventDefault(); togglePlayback();
    }
    else if (event.code === 'ArrowLeft') { event.preventDefault(); step(event.shiftKey ? -10 : -1); }
    else if (event.code === 'ArrowRight') { event.preventDefault(); step(event.shiftKey ? 10 : 1); }
    else if (event.key.toLowerCase() === 'f') { event.preventDefault(); fitWorld(); }
    else if (event.key === 'Escape') { state.selected = null; state.follow = false; $('entity-select').value = ''; renderInspector(); state.lastEventKey = ''; renderEvents(); state.dirty = true; }
  });
  let dragDepth = 0;
  document.addEventListener('dragenter', event => {
    if (!event.dataTransfer?.types.includes('Files')) return;
    event.preventDefault(); dragDepth++; $('drop-overlay').hidden = false;
  });
  document.addEventListener('dragover', event => { if (event.dataTransfer?.types.includes('Files')) { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; } });
  document.addEventListener('dragleave', event => { event.preventDefault(); dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $('drop-overlay').hidden = true; });
  document.addEventListener('drop', event => {
    event.preventDefault(); dragDepth = 0; $('drop-overlay').hidden = true;
    const file = event.dataTransfer?.files[0];
    if (file) openReplay(file, file.name);
  });
  function refreshVisibleManifest() {
    if (!portable && location.protocol !== 'file:' && !document.hidden) loadManifest();
  }
  document.addEventListener('visibilitychange', () => { state.lastTick = null; refreshVisibleManifest(); });
  window.addEventListener('focus', refreshVisibleManifest);
  if (!portable && location.protocol !== 'file:') setInterval(refreshVisibleManifest, 10000);
  $('refresh-recordings').hidden = portable;
  new ResizeObserver(resize).observe($('stage'));
  function animate(timestamp) {
    const elapsed = state.lastTick == null ? 0 : Math.min(0.25, (timestamp - state.lastTick) / 1000);
    state.lastTick = timestamp;
    if (state.playing && state.replay) {
      seek(state.time + elapsed * state.speed);
      if (state.time >= state.replay.frames.at(-1).t) setPlaying(false);
    }
    if (state.dirty) drawWorld();
    requestAnimationFrame(animate);
  }
  resize();
  loadManifest(true);
  requestAnimationFrame(animate);
})();
