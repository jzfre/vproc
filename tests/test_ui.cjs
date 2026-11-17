// Dependency-free UI state regressions: node --test tests/test_ui.cjs
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const app = readFileSync(path.join(__dirname, '../vproc/ui/app.js'), 'utf8');
const memories = ['alpha', 'beta', 'gamma'].map((memory_id) => ({
  memory_id, duration_s: 60, speakers: [memory_id], segment_count: 1,
}));
const segments = (speaker) => [{speaker, start_ts: 0, end_ts: 60, said_text: speaker}];
const response = (body, status = 200) => ({ok: status < 400, status, json: async () => body});
const deferred = () => {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return {promise, resolve};
};

async function viewer() {
  const nodes = new Map();
  const node = (id) => {
    if (!nodes.has(id)) nodes.set(id, {
      innerHTML: '', textContent: '', value: '', disabled: false, hidden: false,
      currentTime: 0, src: '', readyState: 1, error: null, listeners: new Map(),
      style: {setProperty() {}, removeProperty() {}},
      classList: {add() {}, remove() {}, toggle() {}},
      addEventListener(event, listener) { this.listeners.set(event, listener); },
      matches: () => false, querySelectorAll: () => [],
      play: () => Promise.resolve(),
    });
    return nodes.get(id);
  };
  const requests = new Map();
  const timers = {callbacks: [], cleared: []};
  const context = vm.createContext({
    // Controllable timer: a real interval could keep the test process alive.
    setInterval: (callback) => { timers.callbacks.push(callback); return timers.callbacks.length; },
    clearInterval: (id) => { timers.cleared.push(id); },
    AbortController,
    document: {getElementById: node, querySelector: () => null},
    window: {addEventListener() {}}, URLSearchParams,
    location: {search: '', pathname: '/'},
    history: {replaceState(_state, _title, url) { context.url = url; }},
    console,
    fetch: async (url, opts) => {
      if (requests.has(url)) {
        const pending = requests.get(url).promise;
        if (!opts || !opts.signal) return pending;
        return Promise.race([pending, new Promise((_, reject) => {
          opts.signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), {name: 'AbortError'})));
        })]);
      }
      if (url === '/api/memories') return response(memories);
      const match = url.match(/^\/api\/memories\/(\w+)\/segments$/);
      if (match) return response(segments(match[1]));
      throw new Error(`Unexpected request: ${url}`);
    },
  });
  vm.runInContext(app, context);
  // Boot includes both the catalog and first transcript requests.
  await new Promise((resolve) => setImmediate(resolve));
  return {context, node, requests, timers, state: vm.runInContext('state', context)};
}

const delayMeeting = (ui, id) => {
  const request = deferred();
  ui.requests.set(`/api/memories/${id}/segments`, request);
  return request;
};

test('pending meeting switch keeps the displayed media and transcript state together', async () => {
  const ui = await viewer();
  const beta = delayMeeting(ui, 'beta');
  const loading = ui.context.loadMemory('beta');
  assert.equal(ui.state.current, 'alpha');
  assert.equal(ui.context.url, '?memory=alpha');
  assert.equal(ui.node('video').src, '/api/media/alpha');
  beta.resolve(response(segments('beta')));
  await loading;
  assert.equal(ui.state.current, 'beta');
});

test('a slower meeting response cannot overwrite the latest selection', async () => {
  const ui = await viewer();
  const beta = delayMeeting(ui, 'beta');
  const gamma = delayMeeting(ui, 'gamma');
  const loadingBeta = ui.context.loadMemory('beta');
  const loadingGamma = ui.context.loadMemory('gamma');
  gamma.resolve(response(segments('gamma')));
  await loadingGamma;
  beta.resolve(response(segments('beta')));
  await loadingBeta;
  assert.equal(ui.state.current, 'gamma');
  assert.equal(ui.state.segments[0].speaker, 'gamma');
  assert.equal(ui.node('video').src, '/api/media/gamma');
  assert.equal(ui.node('memory-picker').value, 'gamma');
  assert.equal(ui.context.url, '?memory=gamma');
});

test('a failed stale switch cannot roll back a newer successful selection', async () => {
  const ui = await viewer();
  const beta = delayMeeting(ui, 'beta');
  const loadingBeta = ui.context.loadMemory('beta');
  await ui.context.loadMemory('gamma');
  beta.resolve(response({detail: 'gone'}, 404));
  await loadingBeta;
  assert.equal(ui.state.current, 'gamma');
  assert.equal(ui.node('memory-picker').value, 'gamma');
  assert.equal(ui.context.url, '?memory=gamma');
});

test('a failed citation switch preserves playback and visibly explains the failure', async () => {
  const ui = await viewer();
  ui.node('video').currentTime = 7;
  const beta = delayMeeting(ui, 'beta');
  const jump = ui.context.jumpTo({memory_title: 'beta', start_ts: 40});
  beta.resolve(response({detail: 'Meeting was removed'}, 404));
  await jump;
  assert.equal(ui.node('video').currentTime, 7);
  assert.equal(ui.state.current, 'alpha');
  assert.match(ui.node('meeting-status').textContent, /Meeting was removed/);
});

test('a superseded citation cannot seek a later selected meeting', async () => {
  const ui = await viewer();
  const beta = delayMeeting(ui, 'beta');
  const jump = ui.context.jumpTo({memory_title: 'beta', start_ts: 40});
  await ui.context.loadMemory('gamma');
  ui.node('video').currentTime = 7;
  beta.resolve(response(segments('beta')));
  await jump;
  assert.equal(ui.node('video').currentTime, 7);
  assert.equal(ui.node('video').src, '/api/media/gamma');
});

test('a citation to the displayed meeting cancels a pending switch', async () => {
  const ui = await viewer();
  const beta = delayMeeting(ui, 'beta');
  const loading = ui.context.loadMemory('beta');
  await ui.context.jumpTo({memory_title: 'alpha', start_ts: 20});
  beta.resolve(response(segments('beta')));
  await loading;
  assert.equal(ui.state.current, 'alpha');
  assert.equal(ui.node('video').currentTime, 20);
  assert.equal(ui.node('video').src, '/api/media/alpha');
  assert.equal(ui.node('memory-picker').value, 'alpha');
});

test('late media diagnostics do not replace the new meeting fallback', async () => {
  const ui = await viewer();
  const diagnostic = deferred();
  ui.requests.set('/api/media/alpha', diagnostic);
  const oldError = ui.node('video').listeners.get('error')();
  await ui.context.loadMemory('beta');
  const currentFallback = ui.node('video-fallback').innerHTML;
  diagnostic.resolve(response({detail: 'old alpha video missing'}, 404));
  await oldError;
  assert.equal(ui.node('video-fallback').innerHTML, currentFallback);
  assert.equal(ui.node('video-fallback').hidden, true);
});

test('seeking updates the active speaker immediately even with unavailable media', async () => {
  const ui = await viewer();
  ui.state.segments = [{speaker: 'Ada', start_ts: 10, end_ts: 20}];
  ui.context.seek(12);
  assert.match(ui.node('now-speaking').innerHTML, /Ada/);
});

test('a citation seek is reapplied after new media metadata resets playback time', async () => {
  const ui = await viewer();
  const video = ui.node('video');
  video.readyState = 0;
  await ui.context.jumpTo({memory_title: 'beta', start_ts: 24});
  video.currentTime = 0;
  video.readyState = 1;
  video.listeners.get('loadedmetadata')?.();
  assert.equal(video.currentTime, 24);
});

test('switching meetings discards an earlier pending media seek', async () => {
  const ui = await viewer();
  const video = ui.node('video');
  video.readyState = 0;
  await ui.context.jumpTo({memory_title: 'beta', start_ts: 24});
  await ui.context.loadMemory('gamma');
  video.currentTime = 0;
  video.readyState = 1;
  video.listeners.get('loadedmetadata')?.();
  assert.equal(video.currentTime, 0);
});

test('query mode is locked and previous results cleared until the response arrives', async () => {
  const ui = await viewer();
  ui.node('qa-mode').value = 'search';
  ui.node('qa-input').value = 'new query';
  ui.node('qa-results').innerHTML = 'old answer';
  const request = deferred();
  ui.requests.set('/search', request);
  const query = ui.context.runQuery();
  assert.equal(ui.node('qa-mode').disabled, true);
  assert.equal(ui.node('qa-results').innerHTML, '');
  request.resolve(response([]));
  await query;
  assert.equal(ui.node('qa-mode').disabled, false);
  assert.equal(ui.node('qa-go').disabled, false);
  assert.match(ui.node('qa-results').innerHTML, /No matching segments/);
});

test('a long ask shows elapsed time and offers to stop waiting, then cleans up', async () => {
  const ui = await viewer();
  ui.node('qa-mode').value = 'ask';
  ui.node('qa-input').value = 'who decided?';
  ui.node('qa-stop').hidden = true;
  const request = deferred();
  ui.requests.set('/ask', request);
  const query = ui.context.runQuery();
  // The live region is announced once; the number that changes every second is a separate,
  // silent element, or a screen reader would read out every tick and delay the result.
  const announced = ui.node('qa-status').textContent;
  assert.match(announced, /Answering/);
  assert.doesNotMatch(announced, /\d/);
  assert.match(ui.node('qa-elapsed').textContent, /^\d+ s$/);
  assert.equal(ui.node('qa-stop').hidden, false);
  ui.timers.callbacks[0]();  // a tick updates only the counter
  assert.match(ui.node('qa-elapsed').textContent, /^\d+ s$/);
  assert.equal(ui.node('qa-status').textContent, announced);
  request.resolve(response({answered: false, abstained: true, text: 'Not discussed in these meetings.',
    claims: [], evidence: [], abstain_reason: 'unverified'}));
  await query;
  assert.deepEqual(ui.timers.cleared, [1]);
  assert.equal(ui.node('qa-status').textContent, '');
  assert.equal(ui.node('qa-elapsed').textContent, '');
  assert.equal(ui.node('qa-stop').hidden, true);
  assert.equal(ui.node('qa-go').disabled, false);
  // The fixed sentence stays; the cause is explained without showing any unverified text.
  assert.match(ui.node('qa-results').innerHTML, /Not discussed in these meetings/);
  assert.match(ui.node('qa-results').innerHTML, /could not be verified/);
});

test('stop waiting aborts the request, says the server may still finish, and unlocks', async () => {
  const ui = await viewer();
  ui.node('qa-mode').value = 'ask';
  ui.node('qa-input').value = 'who decided?';
  ui.requests.set('/ask', deferred());
  const query = ui.context.runQuery();
  ui.node('qa-stop').listeners.get('click')();
  await query;
  assert.match(ui.node('qa-results').innerHTML, /Stopped waiting/);
  assert.match(ui.node('qa-results').innerHTML, /may still finish/);
  assert.equal(ui.node('qa-go').disabled, false);
  assert.equal(ui.node('qa-input').disabled, false);
});

test('search hits show how near they are and whether they literally match', async () => {
  const ui = await viewer();
  ui.node('qa-mode').value = 'search';
  ui.node('qa-input').value = 'ERR-4127';
  const hit = (extra) => ({memory_title: 'alpha', start_ts: 1, end_ts: 2, speaker: 'alpha', text: 't', ...extra});
  const request = deferred();
  ui.requests.set('/search', request);
  let query = ui.context.runQuery();
  request.resolve(response([hit({similarity: 0.4593, keyword_match: false})]));
  await query;
  assert.match(ui.node('qa-results').innerHTML, /0\.46/);
  assert.match(ui.node('qa-results').innerHTML, /None of these results contains every word/);
  assert.doesNotMatch(ui.node('qa-results').innerHTML, /badge-exact/);

  const second = deferred();
  ui.requests.set('/search', second);
  query = ui.context.runQuery();
  second.resolve(response([hit({similarity: 0.81, keyword_match: true})]));
  await query;
  assert.match(ui.node('qa-results').innerHTML, /badge-exact/);
  assert.doesNotMatch(ui.node('qa-results').innerHTML, /None of these results contains every word/);
});

test('an ask that fails immediately does not claim the server is still working on it', async () => {
  const ui = await viewer();
  ui.node('qa-mode').value = 'ask';
  ui.node('qa-input').value = 'who decided?';
  ui.requests.set('/ask', {promise: Promise.reject(new TypeError('fetch failed'))});
  await ui.context.runQuery();
  assert.match(ui.node('qa-results').innerHTML, /is the server running/);
  assert.doesNotMatch(ui.node('qa-results').innerHTML, /may still be finishing/);
  assert.equal(ui.node('qa-go').disabled, false);
});

test('UI queries stay in the default project shown by the player', async () => {
  const ui = await viewer();
  const bodies = [];
  const fetch = ui.context.fetch;
  ui.context.fetch = async (url, opts) => {
    if (url === '/ask' || url === '/search') {
      bodies.push(JSON.parse(opts.body));
      return response(url === '/ask' ? {answered: false, abstained: true} : []);
    }
    return fetch(url, opts);
  };
  for (const mode of ['ask', 'search']) {
    ui.node('qa-mode').value = mode;
    ui.node('qa-input').value = 'same meeting title exists in another project';
    await ui.context.runQuery();
  }
  assert.deepEqual(bodies.map(body => body.project), ['default', 'default']);
});

test('a stalled search can also stop waiting and unlock the controls', async () => {
  const ui = await viewer();
  ui.node('qa-mode').value = 'search';
  ui.node('qa-input').value = 'release';
  ui.node('qa-stop').hidden = true;
  ui.requests.set('/search', deferred());
  const request = ui.context.runQuery();
  assert.equal(ui.node('qa-stop').hidden, false);
  ui.node('qa-stop').listeners.get('click')();
  await request;
  assert.equal(ui.node('qa-go').disabled, false);
  assert.match(ui.node('qa-results').innerHTML, /Stopped waiting/);
  assert.doesNotMatch(ui.node('qa-results').innerHTML, /queue behind/);
});

test('model text and citation identities render as escaped text', async () => {
  const ui = await viewer();
  const attack = '\"><img src=x onerror=alert(1)>';
  ui.context.renderAskResult({answered: true, claims: [{text: attack, citations: [
    {memory_title: attack, speaker: attack, start_ts: 0}]}]});
  assert.doesNotMatch(ui.node('qa-results').innerHTML, /<img/);
  assert.match(ui.node('qa-results').innerHTML, /&lt;img/);
  ui.context.renderSearchResults([{memory_title: attack, speaker: attack, text: attack, start_ts: 0, end_ts: 1}]);
  assert.doesNotMatch(ui.node('qa-results').innerHTML, /<img/);
  assert.match(ui.node('qa-results').innerHTML, /&quot;/);
});
