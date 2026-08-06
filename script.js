/* script.js — PolyFut Clean (CV key moments + hybrid valuation) */

// GLOBAL STATE
let selectedPosition = null;
let matchStats = [];
let currentScore = { us: 0, them: 0 };
let zoomLevel = 1; let panX = 0; let panY = 0;
let isDragging = false; let startX, startY;
let slowSpeed = 2; let fastSpeed = 8;
let benchBlocks = [];
let aiChatHistory = [];
let isSeeking = false;
let __futMainVideoWired = false;

// --- CV clip library (from server CV pipeline or import) ---
let clipSegmentsLibrary = [];
let activeClipIndex = -1;
let clipWindowStart = null;
let clipWindowEnd = null;
let __rafClampId = null;
let cvJobId = null;
// Optional pitch calibration for THIS video (not stored between uploads).
// null = the user skipped it, and the run behaves exactly as it did before.
let cvPitchCalibration = null;
let cvPollTimer = null;
let cvSegmentsAreDemo = false;
const CV_SERVER_PORTS = [5000, 5050, 8080];
const CV_SESSION_KEY = 'polyfut_cv_session';
const CV_CATALOGUE_KEY = 'polyfut_match_catalogue';
let cvServerBase = '';

// --- NOTIFICATIONS: get the user's attention when input is needed ---------
// Long analyses run in the background, so when PolyFut reaches a step that
// needs the user (pick your team, tap yourself, review touches) we flash the
// taskbar (desktop, via the pywebview bridge) or the title bar (browser) until
// they come back. Only fires when the window is NOT focused, and only if the
// user opted in once (remembered).
const PF_ALERT_PREF_KEY = 'pf_alert_pref';   // 'on' | 'off' | (null = not asked)
const __pfBaseTitle = (typeof document !== 'undefined' && document.title) || 'PolyFut';
let __pfTitleFlash = null;

function pfAlertsEnabled() {
    try { return localStorage.getItem(PF_ALERT_PREF_KEY) === 'on'; } catch (e) { return false; }
}

function pfHasNativeFlash() {
    return !!(window.pywebview && window.pywebview.api && window.pywebview.api.flash);
}

// Flash to signal "your input is needed". No-op if disabled or already focused.
function pfAlertInputNeeded() {
    if (!pfAlertsEnabled()) return;
    if (document.hasFocus && document.hasFocus()) return;   // they're already here
    if (pfHasNativeFlash()) {
        try { window.pywebview.api.flash(); } catch (e) {}
    }
    if (!__pfTitleFlash) {                                   // title-bar fallback
        let on = false;
        __pfTitleFlash = setInterval(function () {
            on = !on;
            document.title = on ? '🔔 Input needed — PolyFut' : __pfBaseTitle;
        }, 900);
    }
}

// Called when the window regains focus — stop nagging.
function pfStopAlert() {
    if (__pfTitleFlash) { clearInterval(__pfTitleFlash); __pfTitleFlash = null; }
    if (typeof document !== 'undefined') document.title = __pfBaseTitle;
    if (pfHasNativeFlash() && window.pywebview.api.unflash) {
        try { window.pywebview.api.unflash(); } catch (e) {}
    }
}
if (typeof window !== 'undefined') window.addEventListener('focus', pfStopAlert);

// Surface pipeline-produced warnings (e.g. "footage is low-resolution, ball
// detection recall will be poor") that the backend already computes but was
// previously dropped on the floor after being read into __v2Montage.warnings.
// A fixed top banner (not nested in a screen's card) survives screen swaps —
// the montage card's innerHTML gets wiped and rebuilt on repeat runs.
// The banner is full-width at top:0, so the "How to use" button collides with it
// in EITHER top corner — on the right it covered the dismiss X, on the left it
// covers the warning text. Publish the banner's height so the button can sit
// below it instead. Measured, not hardcoded: the height depends on how many
// warnings there are and how the text wraps, which changes with window width.
function pfSetWarningHeight(px) {
    document.body.style.setProperty('--pf-warn-h', (px || 0) + 'px');
}

// The app header is a sticky bar at the top of the setup screen, and the
// "How to use" button is fixed at the top-left, so the button sat on top of the
// logo. It only became visible in the light theme: the header is hidden once
// setup-screen is, and on the dark theme that covered most of a session.
//
// Published rather than hardcoded because the header wraps at narrow widths, and
// because it disappears entirely on the later screens - at which point the
// offset has to go back to zero or the button floats in empty space.
function pfSetHeaderHeight() {
    var h = document.querySelector('.app-header');
    var px = 0;
    if (h && getComputedStyle(h).display !== 'none') {
        px = Math.round(h.getBoundingClientRect().height);
    }
    document.body.style.setProperty('--pf-header-h', px + 'px');
}

function pfWatchHeaderHeight() {
    pfSetHeaderHeight();
    var h = document.querySelector('.app-header');
    if (h && window.ResizeObserver) new ResizeObserver(pfSetHeaderHeight).observe(h);
    // The header is hidden by `body:has(#setup-screen.hidden)`, a CSS state
    // change no ResizeObserver reliably reports, so watch the class that
    // drives it as well.
    var setup = document.getElementById('setup-screen');
    if (setup && window.MutationObserver) {
        new MutationObserver(pfSetHeaderHeight)
            .observe(setup, { attributes: true, attributeFilter: ['class'] });
    }
    window.addEventListener('resize', pfSetHeaderHeight);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', pfWatchHeaderHeight);
} else {
    pfWatchHeaderHeight();
}

function pfShowPipelineWarnings(warnings) {
    const list = (warnings || []).filter(Boolean);
    const existing = document.getElementById('pf-pipeline-warnings');
    if (existing) existing.remove();
    if (!list.length) { pfSetWarningHeight(0); return; }
    const wrap = document.createElement('div');
    wrap.id = 'pf-pipeline-warnings';
    wrap.className = 'pf-pipeline-warnings';
    wrap.innerHTML =
        '<div class="pf-pipeline-warnings-body">' +
        list.map(function (w) { return '<p>⚠ ' + w + '</p>'; }).join('') +
        '</div>' +
        '<button type="button" class="pf-pipeline-warnings-close" aria-label="Dismiss">×</button>';
    document.body.appendChild(wrap);
    pfSetWarningHeight(wrap.offsetHeight);
    // Re-publish on resize: a narrower window wraps the text onto more lines and
    // makes the banner taller, which would leave the button overlapping again.
    if (window.ResizeObserver) {
        new ResizeObserver(function () {
            if (wrap.isConnected) pfSetWarningHeight(wrap.offsetHeight);
        }).observe(wrap);
    }
    wrap.querySelector('.pf-pipeline-warnings-close').onclick = function () {
        wrap.remove();
        pfSetWarningHeight(0);
    };
}

// --- update notice -------------------------------------------------------
// Nothing used to tell anyone a newer PolyFut existed, so an install stayed on
// whatever version it started life as and every later fix was invisible to the
// people who already had the app.
//
// The server does the network part (/api/update_check), so this cannot block
// the page or trip over CORS. It only ever informs: no download, no install.
const PF_UPDATE_DISMISS_KEY = 'polyfut_update_dismissed';

function pfShowUpdateNotice(info) {
    if (!info || !info.update_available || !info.latest) return;
    // Dismissal is remembered per VERSION, so saying "later" silences this
    // release but a genuinely newer one still gets to speak up.
    try {
        if (localStorage.getItem(PF_UPDATE_DISMISS_KEY) === info.latest) return;
    } catch (e) { /* private mode: show it, better than swallowing it */ }
    if (document.getElementById('pf-update-notice')) return;

    const wrap = document.createElement('div');
    wrap.id = 'pf-update-notice';
    wrap.className = 'pf-update-notice';
    wrap.setAttribute('role', 'status');
    const url = info.url || 'https://polyfut.com';
    wrap.innerHTML =
        '<span class="pf-update-text">PolyFut <strong>' + info.latest +
        '</strong> is available. You have ' + info.current + '.</span>' +
        '<a class="pf-update-link" target="_blank" rel="noopener" href="' +
        url + '">What&rsquo;s new</a>' +
        '<button type="button" class="pf-update-close" aria-label="Dismiss">' +
        '&times;</button>';
    document.body.appendChild(wrap);
    wrap.querySelector('.pf-update-close').onclick = function () {
        try { localStorage.setItem(PF_UPDATE_DISMISS_KEY, info.latest); } catch (e) { }
        wrap.remove();
    };
}

function pfCheckForUpdate() {
    // Failure is silence. A user opening the app to analyse a match does not
    // need to hear that a version check did not work.
    fetch('/api/update_check')
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(pfShowUpdateNotice)
        .catch(function () { });
}

// Scripts load at the end of <body>, so DOMContentLoaded may already have
// fired by now -- a bare listener would never run and the notice would never
// appear. Check the state instead of assuming.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', pfCheckForUpdate);
} else {
    pfCheckForUpdate();
}

// One-time, remembered opt-in — a small styled prompt (never a native dialog).
function pfMaybeAskAlerts() {
    try { if (localStorage.getItem(PF_ALERT_PREF_KEY) !== null) return; } catch (e) { return; }
    if (document.getElementById('pf-alert-ask')) return;
    const wrap = document.createElement('div');
    wrap.id = 'pf-alert-ask';
    wrap.className = 'pf-alert-ask';
    wrap.innerHTML =
        '<div class="pf-alert-ask-card">' +
        '<div class="pf-alert-ask-title">Notify you when PolyFut needs you?</div>' +
        '<div class="pf-alert-ask-body">Analysis can take a while. PolyFut can flash its taskbar icon when it needs your input — picking your team, tapping yourself, or reviewing touches — so you can do something else and come back when it&rsquo;s ready.</div>' +
        '<div class="pf-alert-ask-actions">' +
        '<button type="button" class="cv-btn-secondary" id="pf-alert-no">Not now</button>' +
        '<button type="button" class="cv-btn-primary" id="pf-alert-yes">Notify me</button>' +
        '</div></div>';
    document.body.appendChild(wrap);
    function choose(pref) {
        try { localStorage.setItem(PF_ALERT_PREF_KEY, pref); } catch (e) {}
        pfSyncAlertToggle();   // keep the setup-screen toggle in step
        wrap.remove();
    }
    document.getElementById('pf-alert-yes').onclick = function () { choose('on'); };
    document.getElementById('pf-alert-no').onclick = function () { choose('off'); };
}

function resolveCvServerBase() {
    if (location.protocol === 'http:' || location.protocol === 'https:') {
        return '';
    }
    const port = (typeof location.port === 'string' && location.port) ? location.port : '5000';
    return 'http://127.0.0.1:' + port;
}

function cvApiUrl(path) {
    return (cvServerBase || resolveCvServerBase()) + path;
}

// Drop a copy of the homepage's drifting neon net inside a container. Waiting
// screens sit above the fixed page background, so they'd otherwise be a static
// black box while work happens. Idempotent — the clone keeps animating.
function __pfNetInto(el) {
    if (!el || el.querySelector('.pf-net-inline')) return;
    const src = document.querySelector('.pf-net-bg');
    if (!src) return;
    const net = src.cloneNode(true);
    net.setAttribute('class', 'pf-net-inline');
    net.setAttribute('aria-hidden', 'true');
    el.insertBefore(net, el.firstChild);
}

function loadCvSession() {
    try {
        var raw = localStorage.getItem(CV_SESSION_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch (e) {
        return null;
    }
}

function saveCvSession(patch) {
    var prev = loadCvSession() || {};
    var next = Object.assign({}, prev, patch || {}, { saved_at: Date.now() });
    try {
        localStorage.setItem(CV_SESSION_KEY, JSON.stringify(next));
    } catch (e) { /* quota */ }
}

function clearCvSession() {
    try {
        localStorage.removeItem(CV_SESSION_KEY);
    } catch (e) { /* ignore */ }
}

function loadMatchCatalogue() {
    try {
        var raw = localStorage.getItem(CV_CATALOGUE_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch (e) {
        return [];
    }
}

function saveMatchCatalogue(entries) {
    try {
        localStorage.setItem(CV_CATALOGUE_KEY, JSON.stringify(entries || []));
    } catch (e) { /* quota */ }
}

function mergeCatalogueEntries(serverList, localList) {
    var byId = {};
    (localList || []).forEach(function (e) {
        if (e && e.job_id) byId[e.job_id] = e;
    });
    (serverList || []).forEach(function (e) {
        if (e && e.job_id) byId[e.job_id] = Object.assign({}, byId[e.job_id] || {}, e);
    });
    return Object.keys(byId).map(function (k) { return byId[k]; }).sort(function (a, b) {
        return (b.analysed_at || 0) - (a.analysed_at || 0);
    });
}

function getSetupMetadataFields() {
    var opp = document.getElementById('opponent-name');
    var su = document.getElementById('score-us');
    var st = document.getElementById('score-them');
    var md = document.getElementById('match-date');
    return {
        opponent: opp ? opp.value : '',
        match_date: md ? md.value : '',
        score_us: su ? parseInt(su.value, 10) || 0 : 0,
        score_them: st ? parseInt(st.value, 10) || 0 : 0,
        position: selectedPosition || ''
    };
}

function pushMatchCatalogueEntry(entry) {
    if (!entry || !entry.job_id) return;
    var list = mergeCatalogueEntries([entry], loadMatchCatalogue());
    saveMatchCatalogue(list);
}

function removeMatchCatalogueEntry(jobId) {
    var list = loadMatchCatalogue().filter(function (e) { return e.job_id !== jobId; });
    saveMatchCatalogue(list);
}

function formatCatalogueDate(ts) {
    if (!ts) return '';
    var d = new Date(ts > 1e12 ? ts : ts * 1000);
    if (isNaN(d.getTime())) return '';
    return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

function formatCatalogueTitle(entry) {
    var opp = (entry.opponent || 'Opponent').toUpperCase();
    var date = entry.match_date ? ' · ' + entry.match_date : '';
    var score = '';
    if (entry.score_us != null && entry.score_them != null) {
        score = ' · ' + entry.score_us + '–' + entry.score_them;
    }
    return 'VS ' + opp + date + score;
}

function sendMatchMetadataToServer(jobId, meta) {
    if (!jobId) return Promise.resolve();
    var body = meta || getSetupMetadataFields();
    return fetch(cvApiUrl('/api/catalogue/' + jobId + '/metadata'), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    }).catch(function () {});
}

function refreshMatchCatalogue() {
    var listEl = document.getElementById('match-catalogue-list');
    var emptyEl = document.getElementById('match-catalogue-empty');
    if (!listEl) return Promise.resolve();

    function render(entries) {
        saveMatchCatalogue(entries);
        listEl.innerHTML = '';
        if (!entries.length) {
            if (emptyEl) emptyEl.classList.remove('hidden');
            return;
        }
        if (emptyEl) emptyEl.classList.add('hidden');
        entries.forEach(function (entry) {
            var card = document.createElement('div');
            card.className = 'match-catalogue-card';
            var title = document.createElement('div');
            title.className = 'match-catalogue-title';
            title.textContent = formatCatalogueTitle(entry);
            var meta = document.createElement('div');
            meta.className = 'match-catalogue-meta';
            var parts = [];
            if (entry.position) parts.push(entry.position);
            parts.push((entry.n_hotspots || 0) + ' hotspots');
            if (entry.n_actions) parts.push(entry.n_actions + ' actions logged');
            else if (entry.has_session) parts.push('Session saved');
            if (entry.analysed_at) parts.push('Analysed ' + formatCatalogueDate(entry.analysed_at));
            if (entry.video_available === false) parts.push('Video missing');
            meta.textContent = parts.join(' · ');
            var actions = document.createElement('div');
            actions.className = 'match-catalogue-actions';
            var openBtn = document.createElement('button');
            openBtn.type = 'button';
            openBtn.className = 'cv-btn-primary';
            openBtn.textContent = 'Open';
            openBtn.onclick = function () { openCatalogueMatch(entry); };
            var rmBtn = document.createElement('button');
            rmBtn.type = 'button';
            rmBtn.className = 'cv-btn-secondary';
            rmBtn.textContent = 'Remove';
            rmBtn.onclick = function () { removeCatalogueMatch(entry.job_id); };
            actions.appendChild(openBtn);
            actions.appendChild(rmBtn);
            card.appendChild(title);
            card.appendChild(meta);
            card.appendChild(actions);
            listEl.appendChild(card);
        });
    }

    return probeCvServer().then(function (ok) {
        if (!ok) {
            render(loadMatchCatalogue());
            return;
        }
        return fetch(cvApiUrl('/api/catalogue'))
            .then(function (r) { return r.json(); })
            .then(function (data) {
                var matches = (data && data.matches) ? data.matches : [];
                render(matches);
            })
            .catch(function () { render(loadMatchCatalogue()); });
    });
}

function openCatalogueMatch(entry) {
    if (!entry || !entry.job_id) return;
    cvJobId = entry.job_id;
    cvToken = entry.token;
    cvMyTeamId = entry.my_team || 'team_a';
    saveCvSession({
        job_id: entry.job_id,
        token: entry.token,
        my_team: cvMyTeamId,
        opponent: entry.opponent,
        match_date: entry.match_date,
        score_us: entry.score_us,
        score_them: entry.score_them,
        position: entry.position,
        state: 'done'
    });
    restoreMatchMetadataFromSession(entry);
    hideCvResumeBanner();
    var setupScreen = document.getElementById('setup-screen');
    if (setupScreen) setupScreen.classList.add('hidden');
    fetch(cvApiUrl('/api/process/status/' + entry.job_id))
        .then(function (r) { return r.json(); })
        .then(function (j) {
            if (j.error) {
                window.alert('Could not open this saved match. It may have been removed.');
                refreshMatchCatalogue();
                return;
            }
            finishFromStatus(j);
        })
        .catch(function () {
            window.alert('Could not reach the server to open this match.');
        });
}

function removeCatalogueMatch(jobId) {
    if (!jobId) return;
    if (!window.confirm('Remove this match from your saved catalogue?')) return;
    removeMatchCatalogueEntry(jobId);
    try { localStorage.removeItem(reviewSessionStorageKey(jobId)); } catch (e) { /* ignore */ }
    fetch(cvApiUrl('/api/catalogue/' + jobId), { method: 'DELETE' }).catch(function () {});
    refreshMatchCatalogue();
}

function reviewSessionStorageKey(jobId) {
    return 'polyfut_review_' + (jobId || '');
}

var __sessionSaveTimer = null;

function serializeMatchSession() {
    return {
        version: 1,
        matchStats: matchStats.slice(),
        currentScore: { us: currentScore.us, them: currentScore.them },
        selectedPosition: selectedPosition,
        benchBlocks: benchBlocks.map(function (b) {
            return { id: b.id, startPct: b.startPct, endPct: b.endPct };
        }),
        activeClipIndex: activeClipIndex,
        clipWindowStart: clipWindowStart,
        clipWindowEnd: clipWindowEnd,
        hybridResults: currentHybridResults,
        updated_at: Date.now()
    };
}

function clearReviewSession() {
    matchStats = [];
    currentHybridResults = null;
    activeClipIndex = -1;
    clipWindowStart = null;
    clipWindowEnd = null;
    clearBenchBlocks();
    refreshLiveDashboard();
}

function clearBenchBlocks() {
    benchBlocks.forEach(function (b) {
        if (b.element) b.element.remove();
    });
    benchBlocks = [];
}

function createBenchBlock(id, startPct, endPct) {
    const track = document.getElementById('bench-track');
    if (!track) return null;
    const newBlock = { id: id, startPct: startPct, endPct: endPct, element: null };
    const blockEl = document.createElement('div');
    blockEl.className = 'bench-block-container';
    blockEl.id = 'block-' + id;
    const fill = document.createElement('div');
    fill.className = 'bench-fill';
    const leftH = document.createElement('div');
    leftH.className = 'bench-handle left';
    const rightH = document.createElement('div');
    rightH.className = 'bench-handle right';
    const closeBtn = document.createElement('div');
    closeBtn.className = 'bench-remove';
    closeBtn.innerText = '×';
    closeBtn.onclick = function () { removeBenchBlock(id); scheduleSaveMatchSession(); };
    blockEl.appendChild(fill);
    blockEl.appendChild(leftH);
    blockEl.appendChild(rightH);
    blockEl.appendChild(closeBtn);
    track.appendChild(blockEl);
    newBlock.element = blockEl;
    setupBlockListeners(newBlock, leftH, rightH);
    renderBlock(newBlock);
    return newBlock;
}

function rebuildBenchBlocksFromSession(blocks) {
    clearBenchBlocks();
    (blocks || []).forEach(function (b) {
        const block = createBenchBlock(
            b.id || Date.now() + Math.floor(Math.random() * 1000),
            typeof b.startPct === 'number' ? b.startPct : 0.4,
            typeof b.endPct === 'number' ? b.endPct : 0.5
        );
        if (block) benchBlocks.push(block);
    });
}

function refreshLiveDashboard() {
    if (typeof calculatePerformance !== 'function') return;
    const videoPlayer = document.getElementById('main-player');
    if (!videoPlayer) return;
    const duration = videoPlayer.duration || 90;
    const liveResults = calculatePerformance(
        matchStats,
        currentScore,
        duration,
        getAllExcludedRanges(duration),
        selectedPosition || 'FW'
    );
    const elNet = document.getElementById('dash-net');
    if (elNet) {
        elNet.innerText = liveResults.netScore;
        elNet.style.color = parseFloat(liveResults.netScore) >= 0 ? '#4caf50' : '#ff2e4d';
    }
    const elOff = document.getElementById('dash-off-markov');
    if (elOff) elOff.innerText = liveResults.offMarkov;
    const elDef = document.getElementById('dash-def-markov');
    if (elDef) elDef.innerText = liveResults.defMarkov;
    const elRisk = document.getElementById('dash-risk');
    if (elRisk) {
        const totalRisk = (parseFloat(liveResults.offRidge) + parseFloat(liveResults.defRidge)).toFixed(3);
        elRisk.innerText = totalRisk;
    }
}

function restoreMatchSession(data) {
    if (!data) return;
    matchStats = Array.isArray(data.matchStats) ? data.matchStats.slice() : [];
    if (data.currentScore) {
        currentScore = {
            us: parseInt(data.currentScore.us, 10) || 0,
            them: parseInt(data.currentScore.them, 10) || 0
        };
        const scoreEl = document.getElementById('display-score');
        if (scoreEl) scoreEl.innerText = currentScore.us + ' - ' + currentScore.them;
    }
    if (data.selectedPosition) {
        selectedPosition = data.selectedPosition;
        document.querySelectorAll('.pitch-zone').forEach(function (el) { el.classList.remove('selected-zone'); });
        const z = document.getElementById('zone' + data.selectedPosition);
        if (z) z.classList.add('selected-zone');
        const disp = document.getElementById('selected-pos-display');
        if (disp) disp.innerText = data.selectedPosition;
    }
    rebuildBenchBlocksFromSession(data.benchBlocks);
    currentHybridResults = data.hybridResults || null;
    clipWindowStart = data.clipWindowStart != null ? data.clipWindowStart : null;
    clipWindowEnd = data.clipWindowEnd != null ? data.clipWindowEnd : null;
    activeClipIndex = typeof data.activeClipIndex === 'number' ? data.activeClipIndex : -1;
    refreshLiveDashboard();
    if (activeClipIndex >= 0 && clipSegmentsLibrary[activeClipIndex]) {
        selectClipSegment(activeClipIndex);
    }
}

function loadMatchSessionForJob(jobId) {
    if (!jobId) return Promise.resolve(null);
    return fetch(cvApiUrl('/api/catalogue/' + jobId + '/session'))
        .then(function (r) { return r.ok ? r.json() : { session: null }; })
        .then(function (data) {
            if (data && data.session) return data.session;
            try {
                var raw = localStorage.getItem(reviewSessionStorageKey(jobId));
                return raw ? JSON.parse(raw) : null;
            } catch (e) {
                return null;
            }
        })
        .catch(function () {
            try {
                var raw = localStorage.getItem(reviewSessionStorageKey(jobId));
                return raw ? JSON.parse(raw) : null;
            } catch (e) {
                return null;
            }
        });
}

function attachSessionRestoreOnVideoReady(jobId) {
    const v = document.getElementById('main-player');
    if (!v || !jobId) return;
    const once = function () {
        v.removeEventListener('loadedmetadata', once);
        loadMatchSessionForJob(jobId).then(function (sess) {
            if (sess) restoreMatchSession(sess);
        });
    };
    if (v.readyState >= 1 && v.duration) once();
    else v.addEventListener('loadedmetadata', once);
}

function saveMatchSession(jobId) {
    if (!jobId) return;
    var payload = serializeMatchSession();
    try {
        localStorage.setItem(reviewSessionStorageKey(jobId), JSON.stringify(payload));
    } catch (e) { /* quota */ }
    fetch(cvApiUrl('/api/catalogue/' + jobId + '/session'), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session: payload })
    }).catch(function () {});
    pushMatchCatalogueEntry({
        job_id: jobId,
        n_actions: matchStats.length,
        has_session: matchStats.length > 0 || benchBlocks.length > 0 || !!currentHybridResults
    });
}

function scheduleSaveMatchSession() {
    if (!cvJobId) return;
    if (__sessionSaveTimer) clearTimeout(__sessionSaveTimer);
    __sessionSaveTimer = setTimeout(function () {
        saveMatchSession(cvJobId);
    }, 800);
}

function cvVideoUrlForToken(token) {
    return token ? cvApiUrl('/api/video/' + token) : null;
}

function restoreMatchMetadataFromSession(sess) {
    if (!sess) return;
    if (sess.position) {
        selectedPosition = sess.position;
        document.querySelectorAll('.pitch-zone').forEach(function (el) { el.classList.remove('selected-zone'); });
        var z = document.getElementById('zone' + sess.position);
        if (z) z.classList.add('selected-zone');
        var disp = document.getElementById('selected-pos-display');
        if (disp) disp.innerText = sess.position;
    }
    if (sess.opponent != null) {
        var opp = document.getElementById('opponent-name');
        if (opp) opp.value = sess.opponent;
    }
    if (sess.match_date != null) {
        var md = document.getElementById('match-date');
        if (md) md.value = sess.match_date;
    }
    if (sess.score_us != null) {
        var su = document.getElementById('score-us');
        if (su) su.value = sess.score_us;
    }
    if (sess.score_them != null) {
        var st = document.getElementById('score-them');
        if (st) st.value = sess.score_them;
    }
    var oppName = (sess.opponent || 'Opponent');
    var matchDateDisplay = sess.match_date ? ' · ' + sess.match_date : '';
    var nameEl = document.getElementById('display-match-name');
    if (nameEl) nameEl.innerText = 'VS ' + oppName.toUpperCase() + matchDateDisplay;
    var scoreEl = document.getElementById('display-score');
    if (scoreEl) scoreEl.innerText = (sess.score_us || 0) + ' - ' + (sess.score_them || 0);
    currentScore = { us: parseInt(sess.score_us, 10) || 0, them: parseInt(sess.score_them, 10) || 0 };
}

function captureSetupMetadataToSession() {
    var oppName = document.getElementById('opponent-name');
    var scoreUs = document.getElementById('score-us');
    var scoreThem = document.getElementById('score-them');
    var matchDate = document.getElementById('match-date');
    saveCvSession({
        opponent: oppName ? oppName.value : '',
        score_us: scoreUs ? scoreUs.value : 0,
        score_them: scoreThem ? scoreThem.value : 0,
        match_date: matchDate ? matchDate.value : '',
        position: selectedPosition
    });
}

function showCvResumeBanner(sess, statusText, resumeLabel) {
    var el = document.getElementById('cv-resume-banner');
    if (!el) return;
    el.classList.remove('hidden');
    var msg = document.getElementById('cv-resume-banner-text');
    if (msg) msg.textContent = statusText || 'An analysis run is in progress.';
    el.dataset.jobId = sess.job_id || '';
    var resumeBtn = document.getElementById('cv-resume-btn');
    if (resumeBtn) resumeBtn.textContent = resumeLabel || 'Resume analysis';
}

function hideCvResumeBanner() {
    var el = document.getElementById('cv-resume-banner');
    if (el) el.classList.add('hidden');
}

function resumeCvAnalysisUi(prefill) {
    var setupScreen = document.getElementById('setup-screen');
    if (setupScreen) setupScreen.classList.add('hidden');
    document.getElementById('cv-team-screen').classList.add('hidden');
    document.getElementById('cv-processing-screen').classList.remove('hidden');
    hideCvResumeBanner();
    showProcessTracker();
    if (prefill) setCvProgress(prefill);
    cvTrackerStartedAt = Date.now() - ((prefill && prefill.elapsed_sec) ? prefill.elapsed_sec * 1000 : 0);
    startTrackerClock();
}

function tryResumeCvSession() {
    if (location.protocol === 'file:') return;
    var sess = loadCvSession();

    function attachAndResume(j, baseSess) {
        var s = Object.assign({}, baseSess || {}, {
            job_id: j.job_id || (baseSess && baseSess.job_id),
            token: j.token || (baseSess && baseSess.token),
            my_team: j.my_team || (baseSess && baseSess.my_team) || 'team_a'
        });
        saveCvSession(s);
        cvJobId = s.job_id;
        cvToken = s.token;
        cvMyTeamId = s.my_team;
        // A resumed run keeps its playing-time window: the server's stored copy
        // is authoritative, but restoring it here keeps the client's fallback
        // copy (sent on /api/v2/process) from silently widening to whole match.
        cvPlayRanges = (j.play_ranges || s.play_ranges || []);
        if (cvToken) cvVideoURL = cvVideoUrlForToken(cvToken);
        restoreMatchMetadataFromSession(s);
        if (j.state === 'running') {
            resumeCvAnalysisUi(j);
            pollCvStatus();
        } else if (j.state === 'done') {
            showCvResumeBanner(
                s,
                'Previous analysis saved — open results or discard to start a new run.',
                'Open results'
            );
        } else if (j.state === 'interrupted') {
            showCvResumeBanner(s, j.error || 'Analysis was interrupted. Discard and start again.');
        } else if (j.state === 'error') {
            showCvResumeBanner(s, j.error || 'Last analysis failed. Discard to start fresh.');
        } else if (j.state === 'cancelled') {
            clearCvSession();
            hideCvResumeBanner();
        }
    }

    function fetchStatus(jobId, baseSess) {
        return fetch(cvApiUrl('/api/process/status/' + jobId))
            .then(function (r) { return r.json(); })
            .then(function (j) {
                if (j.error && j.state === 'unknown') {
                    clearCvSession();
                    return;
                }
                j.job_id = jobId;
                attachAndResume(j, baseSess);
            });
    }

    probeCvServer().then(function (ok) {
        if (!ok) {
            if (sess && sess.job_id) {
                showCvResumeBanner(sess, 'Saved analysis found — start the server to resume.');
            }
            return;
        }
        if (sess && sess.job_id) {
            fetchStatus(sess.job_id, sess).catch(function () {
                showCvResumeBanner(sess, 'Saved analysis found — reconnect to resume progress.');
            });
            return;
        }
        fetch(cvApiUrl('/api/process/active'))
            .then(function (r) { return r.json(); })
            .then(function (data) {
                var runs = (data && data.runs) ? data.runs : [];
                if (runs.length === 1) {
                    var r0 = runs[0];
                    showCvResumeBanner(
                        { job_id: r0.job_id, token: r0.token },
                        (r0.status || 'Analysis in progress') + ' — tap Resume to reconnect.'
                    );
                }
            })
            .catch(function () { /* ignore */ });
    });
}

function resumeCvFromBanner() {
    var sess = loadCvSession();
    var jobId = (sess && sess.job_id) || (document.getElementById('cv-resume-banner') || {}).dataset.jobId;
    if (!jobId) {
        fetch(cvApiUrl('/api/process/active'))
            .then(function (r) { return r.json(); })
            .then(function (data) {
                var runs = (data && data.runs) ? data.runs : [];
                if (runs.length === 1 && runs[0].job_id) {
                    resumeCvFromBannerWithJob(runs[0].job_id, runs[0].token);
                }
            });
        return;
    }
    resumeCvFromBannerWithJob(jobId, sess ? sess.token : null);
}

function resumeCvFromBannerWithJob(jobId, token) {
    cvJobId = jobId;
    cvToken = token || cvToken;
    cvMyTeamId = (loadCvSession() || {}).my_team || 'team_a';
    if (cvToken) cvVideoURL = cvVideoUrlForToken(cvToken);
    restoreMatchMetadataFromSession(loadCvSession());
    fetch(cvApiUrl('/api/process/status/' + cvJobId))
        .then(function (r) { return r.json(); })
        .then(function (j) {
            saveCvSession({
                job_id: cvJobId,
                token: j.token || cvToken,
                my_team: j.my_team || cvMyTeamId,
                state: j.state
            });
            if (j.state === 'done' || j.state === 'review') {
                finishFromStatus(j);
            } else if (j.state === 'running') {
                resumeCvAnalysisUi(j);
                pollCvStatus();
            }
        });
}

function returnToSetupScreen() {
    if (cvJobId) saveMatchSession(cvJobId);
    clipSegmentsLibrary = [];
    cvSegmentsAreDemo = false;
    cvJobId = null;
    releaseClipWindow();

    var app = document.getElementById('app-layout');
    if (app) {
        app.classList.add('hidden');
        app.style.display = '';
    }
    var proc = document.getElementById('cv-processing-screen');
    if (proc) proc.classList.add('hidden');
    var team = document.getElementById('cv-team-screen');
    if (team) team.classList.add('hidden');

    var v = document.getElementById('main-player');
    if (v) {
        v.pause();
        v.removeAttribute('src');
        v.load();
    }
    var placeholder = document.getElementById('vid-placeholder');
    if (placeholder) placeholder.style.display = '';
    var panel = document.getElementById('clip-library-panel');
    if (panel) panel.classList.add('hidden');
    var zones = document.getElementById('seek-zones');
    if (zones) zones.innerHTML = '';

    hideCvResumeBanner();
    if (cvPollTimer) { clearInterval(cvPollTimer); cvPollTimer = null; }
    hideProcessTracker();

    var setupScreen = document.getElementById('setup-screen');
    if (setupScreen) setupScreen.classList.remove('hidden');
    if (typeof checkStartReady === 'function') checkStartReady();
}

function discardCvSession() {
    var sess = loadCvSession();
    var banner = document.getElementById('cv-resume-banner');
    var jobId = (sess && sess.job_id) || (banner && banner.dataset.jobId) || cvJobId || null;
    if (jobId) {
        if (sess && sess.state === 'done') {
            fetch(cvApiUrl('/api/catalogue/' + jobId), { method: 'DELETE' }).catch(function () {});
            removeMatchCatalogueEntry(jobId);
        } else {
            fetch(cvApiUrl('/api/process/' + jobId), { method: 'DELETE' }).catch(function () {});
        }
    }
    clearCvSession();
    cvToken = null;
    returnToSetupScreen();
    refreshMatchCatalogue();
}

function confirmDiscardRun() {
    if (!window.confirm('Discard this analysis run? Hotspots and saved progress will be removed so you can start fresh.')) {
        return;
    }
    discardCvSession();
}

async function probeCvServer() {
    const bases = [];
    if (location.protocol === 'http:' || location.protocol === 'https:') {
        bases.push('');
        // Same-origin probe fails when the page is served by something other than
        // server.py (e.g. VS Code Live Server on :5500) — also try the analyser's
        // known localhost ports. (http only: https pages can't fetch http.)
        if (location.protocol === 'http:' && CV_SERVER_PORTS.indexOf(parseInt(location.port, 10)) === -1) {
            CV_SERVER_PORTS.forEach(function (p) { bases.push('http://127.0.0.1:' + p); });
        }
    } else {
        CV_SERVER_PORTS.forEach(function (p) { bases.push('http://127.0.0.1:' + p); });
    }
    for (let i = 0; i < bases.length; i++) {
        const base = bases[i];
        try {
            const r = await fetch(base + '/api/health', { method: 'GET' });
            if (!r.ok) continue;
            const data = await r.json();
            if (data && data.status === 'ok') {
                cvServerBase = base;
                // region agent log
                __dbgJs('H6', 'script.js:probeCvServer', 'server found', {
                    base: base,
                    protocol: location.protocol,
                    pipeline_ready: !!data.pipeline_ready,
                    fake_cv: !!data.fake_cv
                });
                // endregion
                return data;
            }
        } catch (e) {
            // try next base
        }
    }
    return null;
}

let cvProcessStart = 0;

// region agent log
function __dbgJs(hypothesisId, location, message, data, runId) {
    fetch('http://127.0.0.1:7900/ingest/df7788f7-6d6a-4898-a408-a7fbc948f6ef', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Debug-Session-Id': '9e74f8' },
        body: JSON.stringify({
            sessionId: '9e74f8',
            runId: runId || 'general-audit-v1',
            hypothesisId: hypothesisId,
            location: location,
            message: message,
            data: data || {},
            timestamp: Date.now()
        })
    }).catch(function () {});
}
// endregion

var cvLastLoggedTrackerStage = '';
var cvLastLoggedPollKey = '';

// --- Floating analysis progress tracker (draggable side panel) ---
var CV_PIPELINE_STAGES = [
    { id: 'upload', label: 'Uploading video' },
    { id: 'kits', label: 'Detecting kit colours' },
    { id: 'init', label: 'Starting analysis' },
    { id: 'shot_filter', label: 'Shot filter (stages 1–2)' },
    { id: 'deadtime', label: 'Dead-time filter (stage 3)' },
    { id: 'inference', label: 'Detect & track (stages 4–7)' },
    { id: 'possession', label: 'Possession (stage 8)' },
    { id: 'timestamps', label: 'Touch hotspots (stage 9)' },
    { id: 'done', label: 'Complete' }
];
var cvTrackerVisible = false;
var cvTrackerDismissed = false;
var cvTrackerActiveStage = 'upload';
var cvTrackerStartedAt = 0;
var cvTrackerClockTimer = null;
var cvTrackerLastHeartbeat = 0;
var cvTrackerLastPct = 0;
var cvTrackerCounter = { current: null, total: null, unit: '', loaded: null, totalBytes: null };

function formatRuntime(totalSec) {
    totalSec = Math.max(0, Math.floor(totalSec || 0));
    var h = Math.floor(totalSec / 3600);
    var m = Math.floor((totalSec % 3600) / 60);
    var s = totalSec % 60;
    if (h > 0) {
        return h + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
    }
    return m + ':' + String(s).padStart(2, '0');
}

function formatBytes(n) {
    n = n || 0;
    if (n >= 1048576) return (n / 1048576).toFixed(1) + ' MB';
    if (n >= 1024) return Math.round(n / 1024) + ' KB';
    return n + ' B';
}

function parseStatusCounter(status) {
    if (!status) return null;
    var shotM = status.match(/shot\s*(\d+)\s*\/\s*(\d+)/i);
    if (shotM) return { current: +shotM[1], total: +shotM[2], unit: 'shots' };
    var stepM = status.match(/(\d+)\s*\/\s*(\d+)/);
    if (stepM) return { current: +stepM[1], total: +stepM[2], unit: 'steps' };
    var liveM = status.match(/(\d+)\s+live\s+shot/i);
    if (liveM) return { current: 0, total: +liveM[1], unit: 'shots' };
    return null;
}

function stageStepCounter(stageId) {
    var order = CV_PIPELINE_STAGES.map(function (s) { return s.id; }).filter(function (id) { return id !== 'done'; });
    var idx = order.indexOf(stageId);
    if (idx < 0) return null;
    return { current: idx + 1, total: order.length, unit: 'stages' };
}

function setTrackerCounter(opts) {
    cvTrackerCounter.loaded = null;
    cvTrackerCounter.totalBytes = null;
    if (opts && opts.loaded != null && opts.totalBytes != null) {
        cvTrackerCounter.loaded = opts.loaded;
        cvTrackerCounter.totalBytes = opts.totalBytes;
        cvTrackerCounter.current = null;
        cvTrackerCounter.total = null;
        cvTrackerCounter.unit = '';
        return;
    }
    if (opts && opts.current != null && opts.total != null) {
        cvTrackerCounter.current = opts.current;
        cvTrackerCounter.total = opts.total;
        cvTrackerCounter.unit = opts.unit || '';
        return;
    }
    if (opts && opts.stage) {
        var sc = stageStepCounter(opts.stage);
        if (sc) {
            cvTrackerCounter.current = sc.current;
            cvTrackerCounter.total = sc.total;
            cvTrackerCounter.unit = sc.unit;
        }
    }
}

function startTrackerClock() {
    stopTrackerClock();
    cvTrackerClockTimer = setInterval(tickTrackerClock, 1000);
    tickTrackerClock();
}

function stopTrackerClock() {
    if (cvTrackerClockTimer) {
        clearInterval(cvTrackerClockTimer);
        cvTrackerClockTimer = null;
    }
}

function tickTrackerClock() {
    refreshTrackerRuntimeDisplay();
    if (cvTrackerDismissed && cvTrackerActiveStage !== 'done') {
        var reopen = document.getElementById('cv-process-tracker-reopen');
        if (reopen) reopen.textContent = buildReopenLabel({ progress: cvTrackerLastPct });
    }
}

function refreshTrackerRuntimeDisplay() {
    var runtimeEl = document.getElementById('cv-tracker-runtime');
    var counterEl = document.getElementById('cv-tracker-counter');
    var heartbeatEl = document.getElementById('cv-tracker-heartbeat');
    var barTrack = document.querySelector('.cv-tracker-bar-track');
    var elapsed = cvTrackerStartedAt ? (Date.now() - cvTrackerStartedAt) / 1000 : 0;

    if (runtimeEl) runtimeEl.textContent = formatRuntime(elapsed);

    if (counterEl) {
        var c = cvTrackerCounter;
        if (c.loaded != null && c.totalBytes != null) {
            counterEl.textContent = formatBytes(c.loaded) + ' / ' + formatBytes(c.totalBytes);
        } else if (c.current != null && c.total != null) {
            counterEl.textContent = c.current + ' / ' + c.total +
                (c.unit ? ' ' + c.unit : '');
        } else {
            counterEl.textContent = '— / —';
        }
    }

    if (heartbeatEl) {
        if (cvTrackerLastHeartbeat) {
            var ago = Math.round((Date.now() - cvTrackerLastHeartbeat) / 1000);
            if (ago <= 2) {
                heartbeatEl.textContent = '● live';
                heartbeatEl.className = 'cv-tracker-heartbeat cv-heartbeat-live';
            } else {
                heartbeatEl.textContent = 'updated ' + ago + 's ago';
                heartbeatEl.className = 'cv-tracker-heartbeat';
            }
        } else {
            heartbeatEl.textContent = '● running';
            heartbeatEl.className = 'cv-tracker-heartbeat cv-heartbeat-live';
        }
    }

    if (barTrack) {
        var stalled = cvTrackerLastHeartbeat &&
            (Date.now() - cvTrackerLastHeartbeat) > 8000 &&
            cvTrackerActiveStage !== 'done';
        barTrack.classList.toggle('cv-tracker-stalled', !!stalled);
    }
}

function buildReopenLabel(opts) {
    var pct = Math.round(opts.progress || 0);
    var elapsed = cvTrackerStartedAt ? formatRuntime((Date.now() - cvTrackerStartedAt) / 1000) : '0:00';
    var c = cvTrackerCounter;
    var counter = '—/—';
    if (c.loaded != null && c.totalBytes != null) {
        counter = formatBytes(c.loaded) + '/' + formatBytes(c.totalBytes);
    } else if (c.current != null && c.total != null) {
        counter = c.current + '/' + c.total;
    }
    return counter + ' · ' + elapsed + ' · ' + pct + '%';
}

function initProcessTracker() {
    var panel = document.getElementById('cv-process-tracker');
    var handle = document.getElementById('cv-tracker-drag-handle');
    var closeBtn = document.getElementById('cv-tracker-close');
    var reopen = document.getElementById('cv-process-tracker-reopen');
    var list = document.getElementById('cv-tracker-stages');
    if (!panel || !list) return;

    list.innerHTML = '';
    CV_PIPELINE_STAGES.forEach(function (s) {
        var li = document.createElement('li');
        li.dataset.stage = s.id;
        li.innerHTML = '<span class="cv-stage-icon">○</span><span class="cv-stage-label">' + s.label + '</span>';
        list.appendChild(li);
    });

    try {
        var saved = JSON.parse(sessionStorage.getItem('polyfut_tracker_pos') || 'null');
        if (saved && typeof saved.left === 'number' && typeof saved.top === 'number') {
            panel.style.left = saved.left + 'px';
            panel.style.top = saved.top + 'px';
            panel.style.right = 'auto';
        } else {
            panel.style.top = '100px';
            panel.style.right = '16px';
        }
    } catch (e) {
        panel.style.top = '100px';
        panel.style.right = '16px';
    }

    var dragging = false;
    var dragOffX = 0;
    var dragOffY = 0;

    function onMove(ev) {
        if (!dragging) return;
        var x = (ev.touches ? ev.touches[0].clientX : ev.clientX) - dragOffX;
        var y = (ev.touches ? ev.touches[0].clientY : ev.clientY) - dragOffY;
        x = Math.max(8, Math.min(window.innerWidth - panel.offsetWidth - 8, x));
        y = Math.max(8, Math.min(window.innerHeight - panel.offsetHeight - 8, y));
        panel.style.left = x + 'px';
        panel.style.top = y + 'px';
        panel.style.right = 'auto';
    }

    function onUp() {
        if (!dragging) return;
        dragging = false;
        try {
            sessionStorage.setItem('polyfut_tracker_pos', JSON.stringify({
                left: parseInt(panel.style.left, 10) || 0,
                top: parseInt(panel.style.top, 10) || 0
            }));
        } catch (e) { /* ignore */ }
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        document.removeEventListener('touchmove', onMove);
        document.removeEventListener('touchend', onUp);
    }

    handle.addEventListener('mousedown', function (ev) {
        if (ev.target.closest('.cv-tracker-close')) return;
        dragging = true;
        var rect = panel.getBoundingClientRect();
        dragOffX = ev.clientX - rect.left;
        dragOffY = ev.clientY - rect.top;
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });
    handle.addEventListener('touchstart', function (ev) {
        if (ev.target.closest('.cv-tracker-close')) return;
        dragging = true;
        var rect = panel.getBoundingClientRect();
        dragOffX = ev.touches[0].clientX - rect.left;
        dragOffY = ev.touches[0].clientY - rect.top;
        document.addEventListener('touchmove', onMove, { passive: false });
        document.addEventListener('touchend', onUp);
    }, { passive: true });

    if (closeBtn) {
        closeBtn.addEventListener('click', function (ev) {
            ev.stopPropagation();
            dismissProcessTracker();
        });
    }
    if (reopen) {
        reopen.addEventListener('click', function () {
            cvTrackerDismissed = false;
            showProcessTracker();
        });
    }
    // region agent log
    __dbgJs('B2', 'script.js:initProcessTracker', 'tracker dom init', {
        hasPanel: !!panel,
        hasRuntime: !!document.getElementById('cv-tracker-runtime'),
        hasCounter: !!document.getElementById('cv-tracker-counter'),
        hasHeartbeat: !!document.getElementById('cv-tracker-heartbeat'),
        hasPct: !!document.getElementById('cv-tracker-pct'),
        stageCount: CV_PIPELINE_STAGES.length
    });
    // endregion
}

function showProcessTracker() {
    var panel = document.getElementById('cv-process-tracker');
    var reopen = document.getElementById('cv-process-tracker-reopen');
    if (!panel) return;
    cvTrackerVisible = true;
    cvTrackerDismissed = false;
    panel.classList.remove('hidden');
    if (reopen) reopen.classList.add('hidden');
    if (!cvTrackerStartedAt) cvTrackerStartedAt = Date.now();
    startTrackerClock();
}

function dismissProcessTracker() {
    var panel = document.getElementById('cv-process-tracker');
    var reopen = document.getElementById('cv-process-tracker-reopen');
    cvTrackerDismissed = true;
    cvTrackerVisible = false;
    if (panel) panel.classList.add('hidden');
    if (reopen && cvTrackerActiveStage !== 'done') reopen.classList.remove('hidden');
}

function hideProcessTracker() {
    var panel = document.getElementById('cv-process-tracker');
    var reopen = document.getElementById('cv-process-tracker-reopen');
    cvTrackerVisible = false;
    if (panel) panel.classList.add('hidden');
    if (reopen) reopen.classList.add('hidden');
    cvTrackerStartedAt = 0;
    cvTrackerLastHeartbeat = 0;
    cvTrackerLastPct = 0;
    cvTrackerCounter = { current: null, total: null, unit: '', loaded: null, totalBytes: null };
    stopTrackerClock();
}

function setTrackerStage(stageId, state) {
    cvTrackerActiveStage = stageId;
    var items = document.querySelectorAll('#cv-tracker-stages li');
    var order = CV_PIPELINE_STAGES.map(function (s) { return s.id; });
    var activeIdx = order.indexOf(stageId);
    items.forEach(function (li, i) {
        li.classList.remove('cv-stage-done', 'cv-stage-active', 'cv-stage-error');
        var icon = li.querySelector('.cv-stage-icon');
        if (state === 'error' && li.dataset.stage === stageId) {
            li.classList.add('cv-stage-error');
            if (icon) icon.textContent = '!';
        } else if (i < activeIdx || (stageId === 'done' && i < order.length)) {
            li.classList.add('cv-stage-done');
            if (icon) icon.textContent = '✓';
        } else if (li.dataset.stage === stageId && stageId !== 'done') {
            li.classList.add('cv-stage-active');
            if (icon) icon.textContent = '▸';
        } else if (stageId === 'done' && li.dataset.stage === 'done') {
            li.classList.add('cv-stage-done');
            if (icon) icon.textContent = '✓';
        } else if (icon) {
            icon.textContent = '○';
        }
    });
    if (stageId === 'done') {
        items.forEach(function (li) {
            li.classList.add('cv-stage-done');
            var icon = li.querySelector('.cv-stage-icon');
            if (icon) icon.textContent = '✓';
        });
    }
}

function updateProcessTracker(opts) {
    opts = opts || {};
    if (!cvTrackerVisible && !cvTrackerDismissed && opts.forceShow) showProcessTracker();
    if (opts.stage || opts.progress_current != null || opts.loaded != null || opts.status) {
        cvTrackerLastHeartbeat = Date.now();
    }
    if (opts.progress_current != null && opts.progress_total != null) {
        setTrackerCounter({
            current: opts.progress_current,
            total: opts.progress_total,
            unit: opts.progress_unit || ''
        });
    } else if (opts.loaded != null && opts.totalBytes != null) {
        setTrackerCounter({ loaded: opts.loaded, totalBytes: opts.totalBytes });
    } else if (opts.status) {
        var parsed = parseStatusCounter(opts.status);
        if (parsed) setTrackerCounter(parsed);
        else if (opts.stage) setTrackerCounter({ stage: opts.stage });
    } else if (opts.stage) {
        setTrackerCounter({ stage: opts.stage });
    }
    if (cvTrackerDismissed && cvTrackerActiveStage !== 'done') {
        var reopen = document.getElementById('cv-process-tracker-reopen');
        if (reopen) reopen.textContent = buildReopenLabel(opts);
    }
    if (opts.stage && opts.stage !== cvLastLoggedTrackerStage) {
        cvLastLoggedTrackerStage = opts.stage;
        // region agent log
        __dbgJs('B4', 'script.js:updateProcessTracker', 'tracker stage change', {
            stage: opts.stage,
            progress: opts.progress,
            progress_current: opts.progress_current,
            progress_total: opts.progress_total,
            counter: {
                current: cvTrackerCounter.current,
                total: cvTrackerCounter.total,
                unit: cvTrackerCounter.unit,
                loaded: cvTrackerCounter.loaded,
                totalBytes: cvTrackerCounter.totalBytes
            },
            visible: cvTrackerVisible,
            dismissed: cvTrackerDismissed,
            lastPct: cvTrackerLastPct
        });
        // endregion
    }
    if (!cvTrackerVisible) {
        // region agent log
        if (cvTrackerDismissed && opts.progress != null && opts.progress !== cvTrackerLastPct) {
            __dbgJs('B7', 'script.js:updateProcessTracker', 'dismissed pct not committed', {
                optsProgress: opts.progress,
                lastPct: cvTrackerLastPct
            });
        }
        // endregion
        return;
    }

    var bar = document.getElementById('cv-tracker-bar');
    var st = document.getElementById('cv-tracker-status');
    var meta = document.getElementById('cv-tracker-meta');
    var pctEl = document.getElementById('cv-tracker-pct');
    var pct = Math.max(0, Math.min(100, opts.progress != null ? opts.progress : 0));
    cvTrackerLastPct = pct;

    if (bar) bar.style.width = pct + '%';
    if (pctEl) pctEl.textContent = pct + '%';
    if (st && opts.status) st.textContent = opts.status;
    if (opts.stage) setTrackerStage(opts.stage, opts.state);
    refreshTrackerRuntimeDisplay();

    if (meta) {
        var parts = [];
        var elapsed = opts.elapsed_sec;
        if (elapsed == null && cvTrackerStartedAt) {
            elapsed = Math.round((Date.now() - cvTrackerStartedAt) / 1000);
        }
        if (elapsed != null) parts.push('Runtime ' + formatRuntime(elapsed));
        if (opts.segments_partial && opts.segments_partial.length) {
            parts.push(opts.segments_partial.length + ' hotspot(s) found so far');
        }
        meta.textContent = parts.join(' · ');
    }
}

function uploadVideoForTeams(file, onUploadProgress) {
    return new Promise(function (resolve, reject) {
        var xhr = new XMLHttpRequest();
        var fd = new FormData();
        fd.append('video', file);
        xhr.open('POST', cvApiUrl('/api/teams'));
        xhr.upload.addEventListener('progress', function (ev) {
            if (ev.lengthComputable && onUploadProgress) {
                onUploadProgress(ev.loaded / ev.total, ev.loaded, ev.total);
            }
        });
        xhr.onload = function () {
            try {
                var data = JSON.parse(xhr.responseText);
                if (xhr.status >= 200 && xhr.status < 300 && !data.error) resolve(data);
                else {
                    var e = new Error(data.error || ('HTTP ' + xhr.status));
                    e.invalidVideo = !!data.invalid_video;
                    reject(e);
                }
            } catch (e) {
                reject(e);
            }
        };
        xhr.onerror = function () { reject(new Error('Network error during upload')); };
        xhr.send(fd);
    });
}

// Scoring exclusions are BENCH TIME ONLY. Non-hotspot time must NOT be
// excluded: calculatePerformance drops actions inside excluded ranges, so
// excluding non-possession time silently voided actions logged in free play
// and painted misleading red "BENCH" boxes across the whole results chart.
function getAllExcludedRanges(duration) {
    return benchBlocks.map(function (b) {
        return { start: b.startPct * duration, end: b.endPct * duration };
    });
}

function isClipWindowActive() {
    return clipWindowStart != null && clipWindowEnd != null && clipWindowEnd > clipWindowStart;
}

function formatClock(seconds) {
    const t = Math.max(0, seconds || 0);
    const m = Math.floor(t / 60).toString().padStart(2, '0');
    const s = Math.floor(t % 60).toString().padStart(2, '0');
    return m + ':' + s;
}

// --- 1. CORE CHECK LOGIC ---
// CV-first: only a video is required to start. Position/opponent/score are
// optional metadata used later by the (hidden) logging + scoring screens.
function checkStartReady() {
    const fileInput = document.getElementById('video-input');
    const startBtn = document.getElementById('start-btn');
    const manualBtn = document.getElementById('manual-start-btn');
    if (!fileInput || !startBtn) return;
    const hasVideo = fileInput.files.length > 0;
    if (hasVideo) {
        startBtn.disabled = false;
        startBtn.style.opacity = "1";
        startBtn.style.cursor = "pointer";
        startBtn.innerText = "FIND TOUCH HOTSPOTS";
    } else {
        startBtn.disabled = true;
        startBtn.style.opacity = "0.5";
        startBtn.style.cursor = "not-allowed";
    }
    if (manualBtn) {
        manualBtn.disabled = !hasVideo;
        manualBtn.style.opacity = hasVideo ? "1" : "0.5";
        manualBtn.style.cursor = hasVideo ? "pointer" : "not-allowed";
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const vInput = document.getElementById('video-input');
    if (vInput) vInput.addEventListener('change', checkStartReady);
    pfInitAlertToggle();
});

// Setup-screen toggle to turn the "needs your input" alerts on/off any time.
function pfInitAlertToggle() {
    const t = document.getElementById('pf-alert-toggle');
    if (!t) return;
    t.checked = pfAlertsEnabled();
    t.addEventListener('change', function () {
        try { localStorage.setItem(PF_ALERT_PREF_KEY, t.checked ? 'on' : 'off'); } catch (e) {}
        if (!t.checked) pfStopAlert();   // stop any flash in progress when turning off
    });
}

// Keep the setup toggle in sync when the pref changes elsewhere (the one-time ask).
function pfSyncAlertToggle() {
    const t = document.getElementById('pf-alert-toggle');
    if (t) t.checked = pfAlertsEnabled();
}

// --- 2. POSITION SELECTOR ---
function selectPosition(pos) {
    selectedPosition = pos;
    document.querySelectorAll('.pitch-zone').forEach(el => el.classList.remove('selected-zone'));
    document.getElementById('zone' + pos).classList.add('selected-zone');
    document.getElementById('selected-pos-display').innerText = pos;
    checkStartReady();
}

// --- 3. MAIN APP ---
function enterMainAppWithVideo(fileURL) {
    const app = document.getElementById('app-layout');
    app.classList.remove('hidden');
    app.style.display = 'flex';

    const videoPlayer = document.getElementById('main-player');
    const placeholder = document.getElementById('vid-placeholder');
    const wrapper = document.getElementById('video-wrapper');
    const slider = document.getElementById('seek-slider');

    placeholder.style.display = 'none';
    videoPlayer.src = fileURL;
    videoPlayer.play().catch(function () { console.log('Autoplay blocked'); });

    if (!__futMainVideoWired) {
        __futMainVideoWired = true;
        videoPlayer.addEventListener('timeupdate', updateVideoTimer);
        videoPlayer.addEventListener('loadedmetadata', function () {
            slider.max = videoPlayer.duration;
            renderSeekTicks();
            maybeInitClipLibrary();
        });

        slider.addEventListener('mousedown', function () { isSeeking = true; });
        slider.addEventListener('mouseup', function () { isSeeking = false; });
        slider.addEventListener('input', function () {
            videoPlayer.currentTime = clampClipSeek(parseFloat(slider.value));
        });

        wrapper.addEventListener('wheel', handleWheel, { passive: false });
        wrapper.addEventListener('mousedown', startPan);
        window.addEventListener('mousemove', pan);
        window.addEventListener('mouseup', endPan);
        document.addEventListener('keydown', handleKeyShortcuts);

        // High-frequency playback clamp (rAF) so playback never bleeds past the
        // end of the selected key-moment window (timeupdate fires too coarsely).
        startClipClampLoop();
    }

    updateSpeedConfig();
}

// --- CLIP LIBRARY (TOUCH HOTSPOTS) ---
function clampClipSeek(t) {
    if (!isClipWindowActive()) return t;
    const v = document.getElementById('main-player');
    const dur = v && v.duration ? v.duration : t;
    return Math.min(clipWindowEnd, Math.max(clipWindowStart, Math.min(dur, t)));
}

function startClipClampLoop() {
    if (__rafClampId != null) return;
    const step = function () {
        const v = document.getElementById('main-player');
        if (v && isClipWindowActive() && !isSeeking) {
            if (v.currentTime > clipWindowEnd + 0.03) {
                v.currentTime = clipWindowEnd;
                v.pause();
                const btn = document.getElementById('play-pause-btn');
                if (btn) btn.innerText = '▶';
            } else if (v.currentTime < clipWindowStart - 0.03) {
                v.currentTime = clipWindowStart;
            }
        }
        __rafClampId = requestAnimationFrame(step);
    };
    __rafClampId = requestAnimationFrame(step);
}

function maybeInitClipLibrary() {
    const panel = document.getElementById('clip-library-panel');
    const list = document.getElementById('clip-library-list');
    const cnt = document.getElementById('clip-library-count');
    if (!panel || !list) return;
    if (!clipSegmentsLibrary.length) {
        panel.classList.add('hidden');
        clipWindowStart = null;
        clipWindowEnd = null;
        return;
    }
    panel.classList.remove('hidden');
    if (cnt) cnt.innerText = '(' + clipSegmentsLibrary.length + ')';
    const badge = document.getElementById('clip-demo-badge');
    if (badge) badge.classList.toggle('hidden', !cvSegmentsAreDemo);
    if (cvSegmentsAreDemo && clipSegmentsLibrary.length) {
        console.warn('[PolyFut-CV] DEMO DATA — not from real CV. Install cv/ and disable POLYFUT_FAKE_CV.');
    }
    list.innerHTML = '';
    clipSegmentsLibrary.forEach(function (seg, i) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'clip-library-item' + (i === 0 ? ' active' : '');
        b.innerText = 'Hotspot ' + (i + 1) + '  ' + formatClock(Number(seg.start)) + ' – ' + formatClock(Number(seg.end));
        b.onclick = function () { selectClipSegment(i); };
        list.appendChild(b);
    });
    renderSeekTicks();
    selectClipSegment(0);
}

function renderHotspotZones() {
    const zonesEl = document.getElementById('seek-zones');
    const ticks = document.getElementById('seek-ticks');
    const v = document.getElementById('main-player');
    if (!zonesEl || !ticks || !v) return;
    zonesEl.innerHTML = '';
    ticks.innerHTML = '';
    const dur = v.duration;
    if (!dur || !isFinite(dur)) return;
    clipSegmentsLibrary.forEach(function (seg, idx) {
        const s = Number(seg.start);
        const e = Number(seg.end);
        const leftPct = Math.max(0, Math.min(1, s / dur)) * 100;
        const rightPct = Math.max(0, Math.min(1, e / dur)) * 100;
        const widthPct = Math.max(0.35, rightPct - leftPct);
        const zone = document.createElement('div');
        zone.className = 'seek-zone' + (idx === activeClipIndex ? ' seek-zone-active' : '');
        zone.style.left = leftPct + '%';
        zone.style.width = widthPct + '%';
        zone.title = 'Touch hotspot ' + (idx + 1);
        zonesEl.appendChild(zone);

        const triggers = Array.isArray(seg.action_triggers) ? seg.action_triggers : [];
        const coreMid = seg.core_start != null
            ? (Number(seg.core_start) + Number(seg.core_end != null ? seg.core_end : seg.core_start)) / 2
            : s + (e - s) / 2;
        const points = triggers.length ? triggers : [coreMid];
        points.forEach(function (t) {
            const tick = document.createElement('div');
            tick.className = 'seek-tick';
            tick.style.left = (Math.max(0, Math.min(1, Number(t) / dur)) * 100) + '%';
            ticks.appendChild(tick);
        });
    });
}

function renderSeekTicks() {
    renderHotspotZones();
}

function releaseClipWindow() {
    clipWindowStart = null;
    clipWindowEnd = null;
    activeClipIndex = -1;
    document.querySelectorAll('.clip-library-item').forEach(function (el) { el.classList.remove('active'); });
    renderHotspotZones();
}

function selectClipSegment(i) {
    if (!clipSegmentsLibrary[i]) return;
    activeClipIndex = i;
    let s = Number(clipSegmentsLibrary[i].start);
    let e = Number(clipSegmentsLibrary[i].end);
    const v = document.getElementById('main-player');
    if (v && v.duration && isFinite(v.duration)) {
        e = Math.min(e, v.duration);
        s = Math.min(s, Math.max(0, e - 0.05));
    }
    clipWindowStart = s;
    clipWindowEnd = e;
    document.querySelectorAll('.clip-library-item').forEach(function (el, idx) {
        el.classList.toggle('active', idx === i);
    });
    renderHotspotZones();
    const slider = document.getElementById('seek-slider');
    if (v) {
        v.currentTime = clipWindowStart;
        if (slider) slider.value = clipWindowStart;
        v.play().catch(function () {});
        const btn = document.getElementById('play-pause-btn');
        if (btn) btn.innerText = '⏸';
        updateVideoTimer();
    }
}

// --- 4. START (setup → seed → CV → player) ---
let cvVideoFile = null;
let cvVideoURL = null;

function initApp() {
    const fileInput = document.getElementById('video-input');
    if (!fileInput.files || !fileInput.files[0]) return;

    // Offer notifications once, up front — the whole flow can leave the user
    // waiting, so ask while they're here and remember their choice.
    pfMaybeAskAlerts();

    // Capture optional metadata for the (later) scoring/logging screens.
    const oppName = document.getElementById('opponent-name').value || 'Opponent';
    const scoreUs = parseInt(document.getElementById('score-us').value, 10) || 0;
    const scoreThem = parseInt(document.getElementById('score-them').value, 10) || 0;
    currentScore = { us: scoreUs, them: scoreThem };
    const matchDateVal = document.getElementById('match-date').value;
    const matchDateDisplay = matchDateVal ? ' · ' + matchDateVal : '';
    document.getElementById('display-match-name').innerText = 'VS ' + oppName.toUpperCase() + matchDateDisplay;
    document.getElementById('display-score').innerText = currentScore.us + ' - ' + currentScore.them;

    cvVideoFile = fileInput.files[0];
    cvVideoURL = URL.createObjectURL(cvVideoFile);
    clearReviewSession();
    captureSetupMetadataToSession();

    const setupScreen = document.getElementById('setup-screen');
    if (setupScreen) setupScreen.classList.add('hidden');

    // Ask when they were on the pitch FIRST. The upload + kit detection below
    // run in parallel and take a minute or two anyway, so this step costs no
    // extra wall-clock time — and it means the seed prefetch, which the server
    // now starts from /api/v2/playing_time, never builds a clip for a moment
    // the user wasn't playing.
    showPlayingTimeScreen(cvVideoURL);
    startTeamDetection();
}

// --- Manual bypass: skip the whole CV pipeline (playing-time, team colors,
// seed clips, detection, montage review) and open the tagging screen directly
// on the full raw video. logStat() already supports free-play tagging with no
// hotspots (it only checks bench exclusions, not clipSegmentsLibrary/cvToken),
// so this is the same tagging screen as the AI path — just with an empty
// hotspot library, meaning free scrubbing of the whole video from the start.
// Nothing is uploaded to the server: no token, no job, purely client-side.
function initAppManual() {
    const fileInput = document.getElementById('video-input');
    if (!fileInput.files || !fileInput.files[0]) return;

    const oppName = document.getElementById('opponent-name').value || 'Opponent';
    const scoreUs = parseInt(document.getElementById('score-us').value, 10) || 0;
    const scoreThem = parseInt(document.getElementById('score-them').value, 10) || 0;
    currentScore = { us: scoreUs, them: scoreThem };
    const matchDateVal = document.getElementById('match-date').value;
    const matchDateDisplay = matchDateVal ? ' · ' + matchDateVal : '';
    document.getElementById('display-match-name').innerText = 'VS ' + oppName.toUpperCase() + matchDateDisplay;
    document.getElementById('display-score').innerText = currentScore.us + ' - ' + currentScore.them;

    cvVideoFile = fileInput.files[0];
    cvVideoURL = URL.createObjectURL(cvVideoFile);
    clearReviewSession();
    captureSetupMetadataToSession();

    const setupScreen = document.getElementById('setup-screen');
    if (setupScreen) setupScreen.classList.add('hidden');

    // No job_id/token exist for this session — finishCvAnalysis is told not to
    // add a catalogue card, since Open/Remove both require a job_id and would
    // silently no-op on one, leaving a dead entry in Saved Matches.
    finishCvAnalysis([], 'manual-skip', { skipCatalogue: true });
}

// --- 4a0. PLAYING-TIME WINDOW: when were you actually on the pitch? ---
// Motivation (job e7efd5ac4bc3): on a 111-min video where the user came on at
// 63', the seed clips landed at 11/39/66/94 min and the shuffle roamed to 12
// and 51 min. Tapping "yourself" there seeded the appearance gallery with a
// DIFFERENT player, and every auto-accepted touch was wrong. Declaring the
// on-pitch ranges up front confines the seed clips and the whole pipeline, and
// cuts runtime roughly in proportion to how much of the match was played.
//
// Ranges are [[startSec, endSec], ...] in video time, kept sorted and merged.
// An empty array means "whole match" — the default, and identical to the old
// behaviour end to end.
let cvPlayRanges = [];
let __pfPlayTimeDecided = false;   // has the user finished (or skipped) the step?
let __pfPlayTimeSubmitted = false; // has the decision reached the server?
let __pfPlayTimeDuration = 0;
let __pfPlayTimeDrag = null;       // {rangeIndex, edge} while a handle is held

function pfFmtClock(sec) {
    var t = Math.max(0, Math.round(sec || 0));
    var m = Math.floor(t / 60), s = t % 60;
    return m + ':' + (s < 10 ? '0' : '') + s;
}

function showPlayingTimeScreen(videoURL) {
    var screen = document.getElementById('cv-playtime-screen');
    if (!screen) return;
    cvPlayRanges = [];
    __pfPlayTimeDecided = false;
    __pfPlayTimeSubmitted = false;
    screen.classList.remove('hidden');

    var vid = document.getElementById('cv-playtime-video');
    if (vid && videoURL) {
        vid.src = videoURL;
        // Duration is only known once metadata lands; until then the track has
        // nothing to scale to, so the widget builds itself from this handler.
        var onMeta = function () {
            __pfPlayTimeDuration = (isFinite(vid.duration) && vid.duration > 0) ? vid.duration : 0;
            cvPlayRanges = [[0, __pfPlayTimeDuration]];
            pfRenderPlayingTime();
            pfSeekPlaytimePreview(0);
        };
        if (isFinite(vid.duration) && vid.duration > 0) onMeta();
        else vid.addEventListener('loadedmetadata', onMeta, { once: true });
    }
    pfWirePlayingTimeControls();
    pfAlertInputNeeded();   // this step needs the user before anything can run
}

let __pfPlayTimeWired = false;

function pfWirePlayingTimeControls() {
    if (__pfPlayTimeWired) return;
    __pfPlayTimeWired = true;

    var presets = document.getElementById('cv-playtime-presets');
    if (presets) {
        presets.addEventListener('click', function (ev) {
            var btn = ev.target.closest('[data-preset]');
            if (!btn) return;
            var d = __pfPlayTimeDuration;
            if (btn.dataset.preset === 'whole') cvPlayRanges = [[0, d]];
            else if (btn.dataset.preset === 'first') cvPlayRanges = [[0, d / 2]];
            else cvPlayRanges = [[d / 2, d]];
            pfRenderPlayingTime();
            pfSeekPlaytimePreview(cvPlayRanges[0][0]);
        });
    }

    var add = document.getElementById('cv-playtime-add');
    if (add) add.addEventListener('click', pfAddPlayingPeriod);

    var whole = document.getElementById('cv-playtime-whole');
    if (whole) whole.addEventListener('click', function () {
        cvPlayRanges = [];              // [] == unrestricted, server-side too
        pfFinishPlayingTime();
    });

    var next = document.getElementById('cv-playtime-next');
    if (next) next.addEventListener('click', function () {
        cvPlayRanges = pfMergeRanges(cvPlayRanges);
        pfFinishPlayingTime();
    });

    // Dragging is delegated from the track so handles can be re-rendered freely.
    var track = document.getElementById('cv-playtime-track');
    if (track) {
        track.addEventListener('pointerdown', function (ev) {
            var handle = ev.target.closest('.cv-playtime-handle');
            if (!handle) return;
            __pfPlayTimeDrag = {
                rangeIndex: parseInt(handle.dataset.range, 10),
                edge: handle.dataset.edge
            };
            // Capture keeps the drag alive if the pointer leaves the track; it
            // throws for a pointer that isn't active, which must not kill the drag.
            try { handle.setPointerCapture(ev.pointerId); } catch (e) {}
            ev.preventDefault();
        });
        track.addEventListener('pointermove', function (ev) {
            if (!__pfPlayTimeDrag) return;
            pfDragPlayingHandle(ev);
        });
        var end = function () {
            if (!__pfPlayTimeDrag) return;
            __pfPlayTimeDrag = null;
            // Merge only on release: merging mid-drag would delete the range
            // under the user's finger the moment two periods touched.
            cvPlayRanges = pfMergeRanges(cvPlayRanges);
            pfRenderPlayingTime();
        };
        track.addEventListener('pointerup', end);
        track.addEventListener('pointercancel', end);
    }
}

function pfDragPlayingHandle(ev) {
    var track = document.getElementById('cv-playtime-track');
    var d = __pfPlayTimeDuration;
    if (!track || !d) return;
    var rect = track.getBoundingClientRect();
    var frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / Math.max(1, rect.width)));
    var t = frac * d;
    var r = cvPlayRanges[__pfPlayTimeDrag.rangeIndex];
    if (!r) return;
    // Keep a period at least 10s wide so a handle can't be dragged through its
    // partner and invert the range.
    if (__pfPlayTimeDrag.edge === 'start') r[0] = Math.min(t, r[1] - 10);
    else r[1] = Math.max(t, r[0] + 10);
    r[0] = Math.max(0, r[0]);
    r[1] = Math.min(d, r[1]);
    pfRenderPlayingTime();
    pfSeekPlaytimePreview(t);
}

function pfSeekPlaytimePreview(t) {
    var vid = document.getElementById('cv-playtime-video');
    var label = document.getElementById('cv-playtime-preview-time');
    if (label) label.textContent = pfFmtClock(t);
    if (!vid || !isFinite(t)) return;
    try { vid.currentTime = Math.max(0, Math.min(t, __pfPlayTimeDuration || t)); } catch (e) {}
}

// Sort, clamp and merge touching/overlapping periods — mirrors the server's
// play_ranges.normalize_ranges so what the user sees is what gets analysed.
function pfMergeRanges(ranges) {
    var d = __pfPlayTimeDuration;
    var clean = (ranges || []).map(function (r) {
        return [Math.max(0, Math.min(r[0], r[1])), Math.min(d || r[1], Math.max(r[0], r[1]))];
    }).filter(function (r) { return r[1] - r[0] > 0.5; });
    clean.sort(function (a, b) { return a[0] - b[0]; });
    var out = [];
    clean.forEach(function (r) {
        var last = out[out.length - 1];
        if (last && r[0] <= last[1] + 0.5) last[1] = Math.max(last[1], r[1]);
        else out.push([r[0], r[1]]);
    });
    return out;
}

// Drop a new period into the widest gap between existing ones (or after the
// last), so "Add another period" always produces something visible and grabbable.
function pfAddPlayingPeriod() {
    var d = __pfPlayTimeDuration;
    if (!d) return;
    var sorted = pfMergeRanges(cvPlayRanges);
    var gaps = [];
    var cursor = 0;
    sorted.forEach(function (r) {
        if (r[0] - cursor > 20) gaps.push([cursor, r[0]]);
        cursor = Math.max(cursor, r[1]);
    });
    if (d - cursor > 20) gaps.push([cursor, d]);

    // No gap anywhere — which is the NORMAL case, because the screen opens with
    // the whole video selected. This used to `return` here, so the very first
    // click on "Add another period" did nothing at all and the feature looked
    // like it did not exist. Carve the room out of the widest period instead:
    // someone who came off and back on wants two spells out of one, and that is
    // the only thing this button can sensibly mean when everything is covered.
    if (!gaps.length) {
        if (!sorted.length) {
            cvPlayRanges = [[0, Math.min(d, Math.max(60, d * 0.45))]];
            pfRenderPlayingTime();
            return;
        }
        var wi = 0;
        sorted.forEach(function (r, i) {
            if (r[1] - r[0] > sorted[wi][1] - sorted[wi][0]) wi = i;
        });
        var w = sorted[wi];
        var span = w[1] - w[0];
        if (span < 40) return;              // too short to split into two
        // Leave a real gap in the middle so the two halves cannot merge back
        // together on the next normalise.
        var gap = Math.max(20, Math.min(span * 0.2, 300));
        var half = (span - gap) / 2;
        sorted.splice(wi, 1,
                      [w[0], w[0] + half],
                      [w[1] - half, w[1]]);
        cvPlayRanges = pfMergeRanges(sorted);
        pfRenderPlayingTime();
        return;
    }

    gaps.sort(function (a, b) { return (b[1] - b[0]) - (a[1] - a[0]); });
    var g = gaps[0];
    var gapLen = g[1] - g[0];
    // Never fill the gap edge to edge: a new period that touches its neighbour
    // is merged away on the next normalise, so "Add another period" would
    // silently collapse back to one range instead of adding anything.
    var span = Math.min(gapLen * 0.6, Math.max(60, gapLen * 0.4));
    var mid = (g[0] + g[1]) / 2;
    sorted.push([Math.max(g[0], mid - span / 2), Math.min(g[1], mid + span / 2)]);
    cvPlayRanges = pfMergeRanges(sorted);
    pfRenderPlayingTime();
}

function pfRemovePlayingPeriod(i) {
    if (cvPlayRanges.length <= 1) return;   // never leave the user with nothing
    cvPlayRanges.splice(i, 1);
    pfRenderPlayingTime();
}

function pfRenderPlayingTime() {
    var track = document.getElementById('cv-playtime-track');
    var periods = document.getElementById('cv-playtime-periods');
    var total = document.getElementById('cv-playtime-total');
    var d = __pfPlayTimeDuration;
    if (!track || !d) return;

    // Rebuild the bars, keeping the (static) tick strip.
    Array.prototype.slice.call(track.querySelectorAll('.cv-playtime-range'))
        .forEach(function (el) { el.remove(); });
    pfRenderPlaytimeTicks(d);

    cvPlayRanges.forEach(function (r, i) {
        var bar = document.createElement('div');
        bar.className = 'cv-playtime-range';
        bar.style.left = (r[0] / d * 100) + '%';
        bar.style.width = ((r[1] - r[0]) / d * 100) + '%';

        var label = document.createElement('span');
        label.className = 'cv-playtime-range-label';
        label.textContent = pfFmtClock(r[0]) + ' – ' + pfFmtClock(r[1]);
        bar.appendChild(label);

        ['start', 'end'].forEach(function (edge) {
            var h = document.createElement('button');
            h.type = 'button';
            h.className = 'cv-playtime-handle';
            h.dataset.range = String(i);
            h.dataset.edge = edge;
            h.style.left = edge === 'start' ? '0%' : '100%';
            h.setAttribute('aria-label',
                (edge === 'start' ? 'Start' : 'End') + ' of period ' + (i + 1) +
                ', ' + pfFmtClock(edge === 'start' ? r[0] : r[1]));
            bar.appendChild(h);
        });
        track.appendChild(bar);
    });

    if (periods) {
        periods.innerHTML = '';
        cvPlayRanges.forEach(function (r, i) {
            var row = document.createElement('div');
            row.className = 'cv-playtime-period';
            var name = document.createElement('span');
            name.className = 'cv-playtime-period-name';
            name.textContent = 'Period ' + (i + 1);
            var time = document.createElement('span');
            time.textContent = pfFmtClock(r[0]) + ' → ' + pfFmtClock(r[1]) +
                '  (' + Math.round((r[1] - r[0]) / 60) + ' min)';
            row.appendChild(name);
            row.appendChild(time);
            if (cvPlayRanges.length > 1) {
                var rm = document.createElement('button');
                rm.type = 'button';
                rm.className = 'cv-btn-link';
                rm.textContent = 'Remove';
                rm.onclick = function () { pfRemovePlayingPeriod(i); };
                row.appendChild(rm);
            }
            periods.appendChild(row);
        });
    }

    if (total) {
        var played = cvPlayRanges.reduce(function (a, r) { return a + (r[1] - r[0]); }, 0);
        total.textContent = 'Analysing ' + Math.round(played / 60) + ' of ' +
            Math.round(d / 60) + ' min';
    }

    // Highlight a preset chip only when the window matches it exactly.
    var chips = document.querySelectorAll('#cv-playtime-presets .cv-chip');
    Array.prototype.forEach.call(chips, function (c) {
        var match = false;
        if (cvPlayRanges.length === 1) {
            var r = cvPlayRanges[0];
            if (c.dataset.preset === 'whole') match = r[0] < 1 && r[1] > d - 1;
            else if (c.dataset.preset === 'first') match = r[0] < 1 && Math.abs(r[1] - d / 2) < 1;
            else match = Math.abs(r[0] - d / 2) < 1 && r[1] > d - 1;
        }
        c.classList.toggle('cv-chip-active', match);
    });
}

function pfRenderPlaytimeTicks(d) {
    var ticks = document.getElementById('cv-playtime-ticks');
    if (!ticks || ticks.dataset.forDuration === String(Math.round(d))) return;
    ticks.dataset.forDuration = String(Math.round(d));
    ticks.innerHTML = '';
    // ~8 labelled marks, rounded to a whole number of minutes.
    var stepMin = Math.max(5, Math.round(d / 60 / 8 / 5) * 5);
    for (var m = stepMin; m * 60 < d; m += stepMin) {
        var el = document.createElement('div');
        el.className = 'cv-playtime-tick';
        el.style.left = (m * 60 / d * 100) + '%';
        var lab = document.createElement('span');
        lab.textContent = m + "'";
        el.appendChild(lab);
        ticks.appendChild(el);
    }
}

function pfFinishPlayingTime() {
    __pfPlayTimeDecided = true;
    var screen = document.getElementById('cv-playtime-screen');
    if (screen) screen.classList.add('hidden');
    var vid = document.getElementById('cv-playtime-video');
    if (vid) { try { vid.pause(); vid.removeAttribute('src'); vid.load(); } catch (e) {} }
    saveCvSession({ play_ranges: cvPlayRanges });
    pfSubmitPlayingTime();
    pfShowTeamScreenWhenReady();
}

// Post the window to the server. Needs both a decision and a token, and the two
// arrive in either order (upload runs concurrently with the step), so this is
// called from both sides and no-ops until it can actually send.
function pfSubmitPlayingTime() {
    if (!__pfPlayTimeDecided || !cvToken || __pfPlayTimeSubmitted) return;
    __pfPlayTimeSubmitted = true;
    fetch(cvApiUrl('/api/v2/playing_time'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            token: cvToken,
            ranges: cvPlayRanges,
            my_team_hex: (cvMyTeam && cvMyTeam.hex) || null
        })
    }).then(function (r) { return r.json(); }).then(function (j) {
        // The server normalises (clamps, merges, collapses whole-match to []);
        // adopt its answer so the client and the analysis agree exactly.
        if (j && j.ok && Array.isArray(j.ranges)) {
            cvPlayRanges = j.ranges;
            saveCvSession({ play_ranges: cvPlayRanges });
        }
    }).catch(function () {
        // Offline / server down: the ranges still ride along on /api/v2/process.
        __pfPlayTimeSubmitted = false;
    });
}

function pfShowTeamScreenWhenReady() {
    if (!__pfPlayTimeDecided) return;
    var team = document.getElementById('cv-team-screen');
    if (team) team.classList.remove('hidden');
}

// --- 4a. TEAM COLOUR DETECTION + PICKER ---
let cvToken = null;
let cvTeams = [];
let cvMyTeam = null;
let cvMyTeamId = 'team_a';

function startTeamDetection() {
    const sub = document.getElementById('cv-team-sub');
    const opts = document.getElementById('cv-team-options');
    cvTrackerStartedAt = Date.now();
    showProcessTracker();
    updateProcessTracker({
        forceShow: true,
        stage: 'upload',
        progress: 0,
        status: 'Uploading video for kit detection…'
    });
    // The team screen stays hidden until the playing-time step is done — the two
    // run concurrently, and showing both at once would stack overlays.
    pfShowTeamScreenWhenReady();
    if (sub) sub.innerText = 'Detecting the two kit colours from your video...';
    if (opts) opts.innerHTML = '<div class="cv-team-spinner"></div>';

    if (location.protocol === 'file:') {
        if (sub) {
            sub.innerHTML = '<strong style="color:#ff6b6b;">Do not open index.html directly.</strong> ' +
                'Start the server (<code>python server.py</code>) then open ' +
                '<a href="http://127.0.0.1:5000/" style="color:#7fdfff;">http://127.0.0.1:5000/</a>';
        }
    }

    probeCvServer().then(function (health) {
        if (!health) {
            // region agent log
            __dbgJs('H6', 'script.js:teams', 'server unreachable', {
                protocol: location.protocol,
                href: location.href
            });
            // endregion
            cvToken = null;
            cvTeams = [
                { id: 'team_a', label: 'Team A', hex: '#e23b3b' },
                { id: 'team_b', label: 'Team B', hex: '#e6efe6' }
            ];
            cvSegmentsAreDemo = true;
            const msg = (location.protocol === 'file:')
                ? 'Opened as a local file — start python server.py and use http://127.0.0.1:5000/'
                : 'Cannot reach the analyser. In the project folder run: python server.py';
            updateProcessTracker({
                stage: 'kits',
                progress: 12,
                status: 'Server unreachable — using default kit colours',
                state: 'error'
            });
            renderTeamOptions(msg);
            return;
        }

        updateProcessTracker({
            stage: 'upload',
            progress: 0,
            status: 'Uploading video…'
        });

        return uploadVideoForTeams(cvVideoFile, function (frac, loaded, total) {
            if (frac >= 1) {
                setTrackerCounter({ stage: 'kits' });
                updateProcessTracker({
                    stage: 'kits',
                    progress: 10,
                    status: 'Detecting kit colours from your video…'
                });
            } else {
                updateProcessTracker({
                    stage: 'upload',
                    progress: Math.round(frac * 8),
                    status: 'Uploading video… ' + Math.round(frac * 100) + '%',
                    loaded: loaded,
                    totalBytes: total
                });
            }
        })
            .then(function (data) {
                cvToken = data.token || null;
                cvTeams = data.teams || [];
                cvSegmentsAreDemo = !!data.demo;
                saveCvSession({ token: cvToken, teams: cvTeams });
                // The window may already be decided (upload is slower than the
                // step); this posts it now that there's a token to attach it to.
                pfSubmitPlayingTime();
                // v2: start compiling the soccer model now so the first seed clip
                // is fast by the time the user finishes picking their team.
                if (cvToken) {
                    fetch(cvApiUrl('/api/v2/warm'), {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ token: cvToken }),
                    }).catch(function () {});
                }
                // region agent log
                __dbgJs('H1', 'script.js:teams', 'teams response', {
                    demo: !!data.demo,
                    mode: data.mode,
                    kits_detected: !!data.kits_detected,
                    warning: data.warning || '',
                    teams: (cvTeams || []).map(function (t) { return { id: t.id, hex: t.hex }; }),
                    is_default_red_white: (cvTeams || []).length === 2 &&
                        cvTeams[0].hex === '#e23b3b' && cvTeams[1].hex === '#e6efe6'
                });
                // endregion
                updateProcessTracker({
                    stage: 'kits',
                    progress: 18,
                    status: data.kits_detected
                        ? 'Kit colours detected — pick your team'
                        : 'Using default colours — pick your team'
                });
                renderTeamOptions(data.warning);
            });
    }).catch(function (err) {
        // region agent log
        __dbgJs('H2', 'script.js:teams', 'teams fetch failed', {
            err: String(err && err.message ? err.message : err),
            protocol: location.protocol,
            base: cvServerBase,
            invalidVideo: !!(err && err.invalidVideo)
        });
        // endregion
        if (err && err.invalidVideo) {
            // Empty/corrupt upload — go back to setup, don't run with defaults.
            hideProcessTracker();
            var teamScreen = document.getElementById('cv-team-screen');
            if (teamScreen) teamScreen.classList.add('hidden');
            var setup = document.getElementById('setup-screen');
            if (setup) setup.classList.remove('hidden');
            window.alert(err.message || 'This video could not be read. Please choose a valid video file and try again.');
            return;
        }
        cvToken = null;
        cvTeams = [
            { id: 'team_a', label: 'Team A', hex: '#e23b3b' },
            { id: 'team_b', label: 'Team B', hex: '#e6efe6' }
        ];
        cvSegmentsAreDemo = true;
        const detail = (err && err.message) ? err.message : 'unknown error';
        updateProcessTracker({
            stage: 'kits',
            progress: 12,
            status: 'Kit detection failed — using defaults',
            state: 'error'
        });
        renderTeamOptions('Analyser error: ' + detail + '. Check the server terminal for details.');
    });
}

function renderTeamOptions(warningText) {
    const sub = document.getElementById('cv-team-sub');
    const opts = document.getElementById('cv-team-options');
    if (!opts) return;
    if (sub) {
        if (warningText) {
            sub.innerHTML = '<strong style="color:#ff6b6b;">' + warningText + '</strong>';
        } else if (cvSegmentsAreDemo) {
            sub.innerText = 'Demo mode: pick which side you played for.';
        } else {
            sub.innerText = 'We read these kit colours from your video. Pick your team — '
                + 'and if a kit has more than one colour, tap a circle to change it or + to add one.';
        }
    }
    opts.innerHTML = '';
    // region agent log
    __dbgJs('B1', 'script.js:renderTeamOptions', 'team swatches rendered', {
        n_teams: cvTeams.length,
        hexes: (cvTeams || []).map(function (t) { return t.hex; }),
        demo: cvSegmentsAreDemo,
        warning: warningText || ''
    });
    // endregion
    cvTeams.forEach(function (team, i) {
        opts.appendChild(__cvBuildTeamCard(team, i));
    });
    if (!cvTeams.length && sub) sub.innerText = 'No players detected to read colours from. Try another clip.';
    __cvEyedropInit();      // idempotent; wires the sample-from-video overlay
    pfAlertInputNeeded();   // team picker is ready — needs the user to choose
}

// A kit's colour list. `hexes` is the multi-colour field; fall back to the
// single `hex` for saved sessions and older server responses.
function __cvTeamHexes(team) {
    if (!team) return [];
    if (Array.isArray(team.hexes) && team.hexes.length) return team.hexes.slice();
    return team.hex ? [team.hex] : [];
}

function __cvSetTeamHexes(team, hexes) {
    team.hexes = hexes.slice();
    team.hex = hexes[0] || team.hex;    // dominant colour stays `hex`
}

// One team card: its colour chips (each tappable to re-pick, each removable),
// an "add colour" button, and the row that selects this team. Kits that aren't
// one flat colour need every colour listed — averaging red and blue gives a
// purple the kit doesn't contain, and nothing then matches it.
function __cvBuildTeamCard(team, i) {
    const card = document.createElement('div');
    card.className = 'cv-team-card';

    const chips = document.createElement('div');
    chips.className = 'cv-team-chips';
    const hexes = __cvTeamHexes(team);

    hexes.forEach(function (hx, ci) {
        const wrap = document.createElement('span');
        wrap.className = 'cv-team-chip-wrap';

        const picker = document.createElement('input');
        picker.type = 'color';
        picker.className = 'cv-team-chip-input';
        picker.value = hx;
        picker.title = 'Change this colour';
        picker.setAttribute('aria-label',
            (team.label || 'Team') + ' colour ' + (ci + 1));
        picker.oninput = function () {
            const next = __cvTeamHexes(team);
            next[ci] = picker.value;
            __cvSetTeamHexes(team, next);
            wrap.style.setProperty('--chip', picker.value);
        };
        wrap.style.setProperty('--chip', hx);
        wrap.appendChild(picker);

        if (hexes.length > 1) {
            const del = document.createElement('button');
            del.type = 'button';
            del.className = 'cv-team-chip-del';
            del.innerHTML = '&times;';
            del.title = 'Remove this colour';
            del.setAttribute('aria-label', 'Remove colour ' + (ci + 1));
            del.onclick = function (e) {
                e.stopPropagation();
                const next = __cvTeamHexes(team);
                next.splice(ci, 1);
                __cvSetTeamHexes(team, next);
                renderTeamOptions();
            };
            wrap.appendChild(del);
        }
        chips.appendChild(wrap);
    });

    if (hexes.length < 3) {
        const add = document.createElement('button');
        add.type = 'button';
        add.className = 'cv-team-chip-add';
        add.innerHTML = '+';
        add.title = 'Add another kit colour';
        add.setAttribute('aria-label',
            'Add a colour to ' + (team.label || 'this team'));
        add.onclick = function (e) {
            e.stopPropagation();
            const next = __cvTeamHexes(team);
            next.push(next[next.length - 1] || '#888888');
            __cvSetTeamHexes(team, next);
            renderTeamOptions();
            // Open the picker on the chip we just added.
            const inputs = document.querySelectorAll(
                '.cv-team-card:nth-of-type(' + (i + 1) + ') .cv-team-chip-input');
            const last = inputs[inputs.length - 1];
            if (last && last.click) last.click();
        };
        chips.appendChild(add);
    }

    // Sample this kit's main colour off the video. Automatic detection reads a
    // torso crop that is mostly pitch on wide footage, so it can report the
    // ground rather than the shirt; clicking the shirt sidesteps that. It also
    // beats typing a hex, because what matters downstream is the colour AS THE
    // CAMERA SEES IT — a "true" navy would never match one the exposure has
    // crushed toward black.
    if (cvToken) {
        const drop = document.createElement('button');
        drop.type = 'button';
        drop.className = 'cv-team-chip-pipette';
        drop.innerHTML = '&#128167;';
        drop.title = 'Pick this team’s main colour from the video';
        drop.setAttribute('aria-label',
            'Pick ' + (team.label || 'this team') + '’s colour from the video');
        drop.onclick = function (e) {
            e.stopPropagation();
            __cvOpenEyedropper(i);
        };
        chips.appendChild(drop);
    }

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'cv-team-swatch';
    const label = document.createElement('span');
    label.className = 'cv-team-label';
    label.innerText = team.label || (team.id === 'team_b' ? 'Team B' : 'Team A');
    btn.appendChild(label);
    const pick = document.createElement('span');
    pick.className = 'cv-team-pick';
    pick.innerText = 'This is my team';
    btn.appendChild(pick);
    btn.onclick = function () { pickTeam(i); };

    card.appendChild(chips);
    card.appendChild(btn);
    return card;
}

// --- eye-dropper: take a kit colour straight off a video frame -------------
// Frames are fetched once and reused; the endpoint returns them at native
// resolution so the shirt is as sharp as the source allows.
let __cvEyedrop = {
    teamIndex: 0, frames: [], cur: 0, img: null, hex: null, loaded: false,
};

function __cvEyedropEls() {
    return {
        screen: document.getElementById('cv-eyedrop-screen'),
        canvas: document.getElementById('cv-eyedrop-canvas'),
        zoom: document.getElementById('cv-eyedrop-zoom'),
        loupe: document.getElementById('cv-eyedrop-loupe'),
        tabs: document.getElementById('cv-eyedrop-tabs'),
        preview: document.getElementById('cv-eyedrop-preview'),
        readout: document.getElementById('cv-eyedrop-readout'),
        use: document.getElementById('cv-eyedrop-use'),
        sub: document.getElementById('cv-eyedrop-sub'),
    };
}

function __cvOpenEyedropper(teamIndex) {
    const el = __cvEyedropEls();
    if (!el.screen || !cvToken) return;
    __cvEyedrop.teamIndex = teamIndex;
    __cvEyedrop.hex = null;
    el.use.disabled = true;
    el.preview.style.background = '#11141b';
    el.readout.innerText = 'Hover the video to preview a colour.';
    el.screen.classList.remove('hidden');
    if (__cvEyedrop.loaded && __cvEyedrop.frames.length) {
        __cvEyedropShow(__cvEyedrop.cur);
        return;
    }
    el.sub.innerText = 'Loading frames from your video…';
    const fd = new FormData();
    fd.append('token', cvToken);
    fd.append('count', '6');
    fetch('/api/v2/calibration_frames', { method: 'POST', body: fd })
        .then(function (r) { return r.json(); })
        .then(function (j) {
            if (!j || !j.frames || !j.frames.length) {
                el.sub.innerText = 'Could not read frames from this video.';
                return;
            }
            __cvEyedrop.frames = j.frames;
            __cvEyedrop.loaded = true;
            __cvEyedrop.cur = Math.min(1, j.frames.length - 1);
            el.sub.innerText = 'Click a player wearing your kit. Pick the body, '
                + 'not the shorts or a shadow.';
            __cvEyedropTabs();
            __cvEyedropShow(__cvEyedrop.cur);
        })
        .catch(function () {
            el.sub.innerText = 'Could not reach the server for video frames.';
        });
}

function __cvEyedropTabs() {
    const el = __cvEyedropEls();
    if (!el.tabs) return;
    el.tabs.innerHTML = '';
    __cvEyedrop.frames.forEach(function (fr, i) {
        const b = document.createElement('button');
        b.type = 'button';
        b.innerText = Math.round(fr.t_sec) + 's';
        b.setAttribute('aria-current', i === __cvEyedrop.cur ? 'true' : 'false');
        b.onclick = function () { __cvEyedropShow(i); };
        el.tabs.appendChild(b);
    });
}

function __cvEyedropShow(i) {
    const el = __cvEyedropEls();
    const fr = __cvEyedrop.frames[i];
    if (!fr || !el.canvas) return;
    __cvEyedrop.cur = i;
    __cvEyedropTabs();
    const img = new Image();
    img.onload = function () {
        el.canvas.width = img.naturalWidth;
        el.canvas.height = img.naturalHeight;
        el.canvas.getContext('2d').drawImage(img, 0, 0);
        __cvEyedrop.img = img;
    };
    img.src = 'data:image/jpeg;base64,' + fr.jpeg_b64;
}

// Canvas pixel under a pointer event. The canvas is CSS-scaled to fit, so the
// ratio between its backing store and its displayed box is what maps a click
// to a pixel — using clientWidth alone would sample the wrong place.
function __cvEyedropPixel(ev) {
    const el = __cvEyedropEls();
    const c = el.canvas;
    const r = c.getBoundingClientRect();
    if (!r.width || !r.height) return null;
    const x = Math.floor((ev.clientX - r.left) * (c.width / r.width));
    const y = Math.floor((ev.clientY - r.top) * (c.height / r.height));
    if (x < 0 || y < 0 || x >= c.width || y >= c.height) return null;
    return { x: x, y: y };
}

// Median of a small neighbourhood, not one pixel: a single pixel on a ~10px
// shirt is as likely to be a compression artefact or a blown highlight as the
// kit, and a median is unmoved by either.
function __cvEyedropSample(x, y, half) {
    const c = __cvEyedropEls().canvas;
    half = half || 1;
    const x0 = Math.max(0, x - half), y0 = Math.max(0, y - half);
    const x1 = Math.min(c.width - 1, x + half), y1 = Math.min(c.height - 1, y + half);
    const w = x1 - x0 + 1, h = y1 - y0 + 1;
    if (w < 1 || h < 1) return null;
    const d = c.getContext('2d').getImageData(x0, y0, w, h).data;
    const rs = [], gs = [], bs = [];
    for (let i = 0; i < d.length; i += 4) { rs.push(d[i]); gs.push(d[i+1]); bs.push(d[i+2]); }
    const med = function (a) { a.sort(function (p, q) { return p - q; });
                               return a[Math.floor(a.length / 2)]; };
    return { r: med(rs), g: med(gs), b: med(bs) };
}

function __cvRgbToHex(c) {
    const h = function (v) { return ('0' + Math.max(0, Math.min(255, v | 0)).toString(16)).slice(-2); };
    return '#' + h(c.r) + h(c.g) + h(c.b);
}

function __cvEyedropInit() {
    const el = __cvEyedropEls();
    if (!el.canvas || el.canvas.__wired) return;
    el.canvas.__wired = true;

    el.canvas.addEventListener('mousemove', function (ev) {
        const p = __cvEyedropPixel(ev);
        if (!p) return;
        const s = __cvEyedropSample(p.x, p.y, 1);
        if (!s) return;
        const hex = __cvRgbToHex(s);
        el.preview.style.background = hex;
        el.readout.innerText = hex + '  —  click to take it';
        // Magnifier, so a ~10px shirt is actually aimable. Sits just above the
        // cursor when there is room and just below when there isn't, and is
        // clamped inside the stage — a fixed offset flies off the top on a
        // short frame and the loupe ends up parked in the corner.
        const stage = el.canvas.parentElement.getBoundingClientRect();
        const LO = 120, GAP = 14;
        const cxr = ev.clientX - stage.left, cyr = ev.clientY - stage.top;
        let ly = cyr - LO - GAP;
        if (ly < 0) ly = Math.min(cyr + GAP, stage.height - LO);
        const lx = Math.max(0, Math.min(cxr - LO / 2, stage.width - LO));
        el.loupe.classList.remove('hidden');
        el.loupe.style.left = Math.max(0, lx) + 'px';
        el.loupe.style.top = Math.max(0, ly) + 'px';
        const z = el.zoom.getContext('2d');
        z.imageSmoothingEnabled = false;
        z.clearRect(0, 0, 120, 120);
        if (__cvEyedrop.img) {
            z.drawImage(__cvEyedrop.img, p.x - 6, p.y - 6, 13, 13, 0, 0, 120, 120);
            z.strokeStyle = '#2ecc71';
            z.lineWidth = 2;
            z.strokeRect(51, 51, 18, 18);
        }
    });
    el.canvas.addEventListener('mouseleave', function () {
        el.loupe.classList.add('hidden');
    });
    el.canvas.addEventListener('click', function (ev) {
        const p = __cvEyedropPixel(ev);
        if (!p) return;
        const s = __cvEyedropSample(p.x, p.y, 1);
        if (!s) return;
        __cvEyedrop.hex = __cvRgbToHex(s);
        el.preview.style.background = __cvEyedrop.hex;
        el.readout.innerText = 'Picked ' + __cvEyedrop.hex
            + ' — press "Use this colour", or click again to re-pick.';
        el.use.disabled = false;
    });

    const cancel = document.getElementById('cv-eyedrop-cancel');
    if (cancel) cancel.onclick = function () {
        el.screen.classList.add('hidden');
    };
    if (el.use) el.use.onclick = function () {
        if (!__cvEyedrop.hex) return;
        const team = cvTeams[__cvEyedrop.teamIndex];
        if (team) {
            const next = __cvTeamHexes(team);
            if (next.length) next[0] = __cvEyedrop.hex;
            else next.push(__cvEyedrop.hex);
            __cvSetTeamHexes(team, next);
            team.hex_source = 'eyedropper';
        }
        el.screen.classList.add('hidden');
        renderTeamOptions();
    };
}

function pickTeam(i) {
    if (cvSegmentsAreDemo) {
        const proceed = window.confirm(
            'Demo mode: results are fake sample clips (first ~10 min only), not from your video.\n\n' +
            'For real analysis: run "pip install -r requirements.txt" in the PolyFut folder, then restart the server (python server.py) without POLYFUT_FAKE_CV.\n\n' +
            'Continue with demo anyway?'
        );
        if (!proceed) return;
    }
    cvMyTeam = cvTeams[i] || null;
    cvMyTeamId = (cvMyTeam && cvMyTeam.id) ? cvMyTeam.id : (i === 0 ? 'team_a' : 'team_b');
    document.getElementById('cv-team-screen').classList.add('hidden');
    if (!cvToken) {
        // Offline / server unreachable: no token to analyse against, so fall
        // back to the in-browser sample demo.
        runBrowserDemo();
        return;
    }
    // Optional: mark out the pitch, so "who touched the ball" can be judged in
    // metres rather than pixels. Skipping leaves everything exactly as before.
    showPitchCalibScreen(cvToken, cvPlayRanges, function (calibration) {
        cvPitchCalibration = calibration || null;
        // v2: identify YOU (not just the team) via a few tapped frames.
        var seedVid = document.getElementById('cv-seed-video');
        var dur = (seedVid && isFinite(seedVid.duration) && seedVid.duration > 0)
            ? seedVid.duration : 0;
        showV2SeedScreen(cvVideoURL, dur);
    });
}

// --- 4a. v2 SEED SCREEN: tap yourself on a tracked node in a 3s clip ---
// Each detected player gets a marker that follows them; tapping one turns that
// player's whole-clip track into seed taps [{t_sec, nx, ny}] for /api/v2/process.
let __v2Seed = null;
let __pfSeedReadyAlerted = false;   // fire the "seed ready" alert only once per session
let __v2SeedRAF = null;
let __v2SeedBuilt = false;      // has any clip finished building this session?
let __v2SeedLoadTimer = null;   // live elapsed-time ticker while a clip builds

// --- team-colour filter: show only your team's markers (keep unknown shown) ---
// Convert a #rrggbb kit colour to OpenCV-style HSV ([H 0-179, S/V 0-255]) so it
// compares against the backend's per-tracklet kit_hsv.
function __hexToHsv(hex) {
    hex = String(hex || '').replace('#', '');
    if (hex.length === 3) hex = hex.split('').map(function (c) { return c + c; }).join('');
    if (hex.length < 6) return null;
    var r = parseInt(hex.substr(0, 2), 16) / 255,
        g = parseInt(hex.substr(2, 2), 16) / 255,
        b = parseInt(hex.substr(4, 2), 16) / 255;
    if (isNaN(r) || isNaN(g) || isNaN(b)) return null;
    var mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn, hh = 0;
    if (d !== 0) {
        if (mx === r) hh = ((g - b) / d) % 6;
        else if (mx === g) hh = (b - r) / d + 2;
        else hh = (r - g) / d + 4;
        hh *= 60; if (hh < 0) hh += 360;
    }
    return [hh / 2, (mx === 0 ? 0 : d / mx) * 255, mx * 255];
}

// Hue-weighted circular HSV distance (mirrors backend color.hsv_distance).
function __hsvDist(a, b) {
    if (!a || !b) return null;
    var dh = Math.abs(a[0] - b[0]); dh = Math.min(dh, 180 - dh);
    var ds = Math.abs(a[1] - b[1]), dv = Math.abs(a[2] - b[2]);
    return Math.sqrt((2 * dh) * (2 * dh) + ds * ds + (0.5 * dv) * (0.5 * dv));
}

function __v2SeedDefaultKits() {
    // The fallback red/white pair means kit detection failed → don't filter.
    return (cvTeams || []).length === 2 &&
        cvTeams[0].hex === '#e23b3b' && cvTeams[1].hex === '#e6efe6';
}

// Split a clip's players into two kit colours and hide the team you didn't pick.
// Rather than testing each player against the detected team hex with an absolute
// threshold (which let black players far from *both* detected colours slip
// through), we cluster the players actually in the clip into two colours, then
// keep the cluster nearest your selected kit and hide the other outright.
// Returns a boolean[] (hidden) aligned to `tracklets`. Unknown-colour players are
// left shown so you're never accidentally hidden.
function __v2SeedTeamHiddenFlags(tracklets) {
    var n = tracklets.length, hidden = new Array(n).fill(false);
    if (!cvMyTeam || !cvMyTeam.hex || __v2SeedDefaultKits()) return hidden;
    var mine = __hexToHsv(cvMyTeam.hex);
    if (!mine) return hidden;

    var idx = [], cols = [];
    for (var i = 0; i < n; i++) {
        if (tracklets[i].kit_hsv) { idx.push(i); cols.push(tracklets[i].kit_hsv); }
    }
    if (cols.length < 3) return hidden;              // too few to trust a 2-way split

    // Seed 2-means with the two most different colours, then a few iterations.
    var a = 0, b = 1, bestD = -1;
    for (var p = 0; p < cols.length; p++) {
        for (var q = p + 1; q < cols.length; q++) {
            var d = __hsvDist(cols[p], cols[q]);
            if (d != null && d > bestD) { bestD = d; a = p; b = q; }
        }
    }
    var c0 = cols[a].slice(), c1 = cols[b].slice(), assign = new Array(cols.length);
    for (var it = 0; it < 4; it++) {
        var s0 = [0, 0, 0], n0 = 0, s1 = [0, 0, 0], n1 = 0;
        for (var k = 0; k < cols.length; k++) {
            var g = (__hsvDist(cols[k], c0) <= __hsvDist(cols[k], c1)) ? 0 : 1;
            assign[k] = g;
            var s = (g === 0) ? s0 : s1;
            s[0] += cols[k][0]; s[1] += cols[k][1]; s[2] += cols[k][2];
            if (g === 0) n0++; else n1++;
        }
        if (n0) c0 = [s0[0] / n0, s0[1] / n0, s0[2] / n0];
        if (n1) c1 = [s1[0] / n1, s1[1] / n1, s1[2] / n1];
    }
    // If the two clusters aren't clearly different, it's really one team on
    // screen — don't split it in half.
    if (__hsvDist(c0, c1) < 45) return hidden;

    // ...and that distance alone is not enough, because it counts saturation
    // and brightness. Measured on a real clip (f1fcfbb84a0d @21s, 14 tracked
    // players): every kit read landed at hue 17-26 — a spread of 8.5, sitting
    // on that pitch's own turf hue of 28 — while saturation spread 192. The
    // grass mask in the backend only catches 1.2% of that (bleached) turf, so
    // jersey_hsv was reporting the GROUND for every player, and the two
    // "kits" the split found differed in hue by 0.6. It was separating players
    // by how much their shirt lightened the grass behind it, then hiding
    // whichever half you had not picked: 6 of 14 players vanished, including
    // the user's own.
    //
    // Two different kits differ in HUE, or one of them is unsaturated (a white
    // or grey shirt against a coloured one). Same hue + different brightness is
    // one surface under two lightings, never two teams. Getting this wrong
    // hides the player from the screen whose entire purpose is finding them, so
    // when the colour cannot be trusted, nothing is hidden.
    var dHue = Math.abs(c0[0] - c1[0]);
    dHue = Math.min(dHue, 180 - dHue);
    var oneIsUnsaturated = (Math.min(c0[1], c1[1]) < 60 &&
                            Math.max(c0[1], c1[1]) >= 100);
    if (dHue < 12 && !oneIsUnsaturated) return hidden;

    var mineCluster = (__hsvDist(c0, mine) <= __hsvDist(c1, mine)) ? 0 : 1;
    for (var k2 = 0; k2 < cols.length; k2++) {
        if (assign[k2] !== mineCluster) hidden[idx[k2]] = true;
    }
    return hidden;
}

function showV2SeedScreen(url, duration) {
    __pfSeedReadyAlerted = false;   // arm the one-shot "seed ready" alert
    __v2Seed = {
        // Per-clip reroll: shuffling reshuffles only the current slot, so a
        // good clip you've already marked isn't disturbed by fixing a bad one.
        slotReroll: [0, 0, 0, 0], index: 0, moments: [], nMoments: 4,
        clip: null, cache: {},           // cache key `${slotReroll[i]}_${index}` -> clip
        chosen: {},                      // index -> chosen t_center, so clips avoid each other
        shownMoments: {},                // index -> [t_centers shown], so shuffle never repeats
        // Mark yourself in as many clips as you appear in — every clip's taps
        // are combined into a stronger appearance seed. key -> {trackId, taps}.
        selections: {},
        showAll: false,          // reveal other-team markers (safety toggle)
        teamFiltered: false,
    };
    var chk = document.getElementById('cv-seed-showall');
    if (chk) chk.checked = false;
    document.getElementById('cv-seed-screen').classList.remove('hidden');
    __v2SeedLoadIndex(function () { __v2SeedLoadClip(0); });
}

// Show the "show all players" toggle only when the team filter actually hid
// someone (and reflect current state).
function __v2SeedUpdateShowAll() {
    const s = __v2Seed;
    const wrap = document.getElementById('cv-seed-showall-wrap');
    if (!wrap || !s) return;
    wrap.classList.toggle('hidden', !s.teamFiltered && !s.showAll);
}

function __v2SeedToggleShowAll(on) {
    const s = __v2Seed;
    if (s) s.showAll = !!on;   // anim loop reads this live
}

function __v2SeedSelCount() {
    const s = __v2Seed;
    return (s && s.selections) ? Object.keys(s.selections).length : 0;
}

function __v2SeedCombinedTaps() {
    const s = __v2Seed;
    if (!s || !s.selections) return [];
    let out = [];
    Object.keys(s.selections).forEach(function (k) {
        out = out.concat(s.selections[k].taps || []);
    });
    return out;
}

// Enable "Find my touches" once you've marked yourself in ≥1 clip; the label
// shows how many clips you've marked so far.
function __v2SeedUpdateFinishBtn() {
    const n = __v2SeedSelCount();
    const btn = document.getElementById('cv-seed-next');
    if (!btn) return;
    btn.disabled = n === 0;
    btn.innerText = n > 0
        ? 'Find My Touches (' + n + ')' + (n === 1 ? ' clip' : ' clips')
        : 'Find My Touches';
}

function __v2SeedApi(path) { return cvApiUrl(path); }

// Moments a newly-built clip should steer clear of: the OTHER clips' current
// moments (so two clips never share a passage of play) PLUS this slot's own
// shuffle history (so shuffling gives a genuinely new clip, never a repeat).
function __v2SeedAvoidFor(index) {
    const s = __v2Seed;
    if (!s) return [];
    const out = [];
    Object.keys(s.chosen || {}).forEach(function (k) {
        if (+k !== index && typeof s.chosen[k] === 'number') out.push(s.chosen[k]);
    });
    ((s.shownMoments || {})[index] || []).forEach(function (m) {
        if (typeof m === 'number') out.push(m);
    });
    return out;
}

// Record a freshly-built clip's moment in the slot's history so future shuffles
// avoid it. If the backend returned a moment we've already shown (it couldn't
// find a new qualifying one — the well is dry), recycle quietly: restart this
// slot's history so shuffling keeps working from the best moments again.
function __v2SeedRecordMoment(index, t) {
    const s = __v2Seed;
    if (!s || typeof t !== 'number') return;
    if (!s.shownMoments) s.shownMoments = {};
    var hist = s.shownMoments[index] || [];
    var repeat = hist.some(function (m) { return Math.abs(m - t) < 8; });
    s.shownMoments[index] = repeat ? [t] : hist.concat([t]).slice(-20);
}

function __v2SeedLoadIndex(then) {
    const s = __v2Seed;
    if (!s || !cvToken) { if (then) then(); return; }
    fetch(__v2SeedApi('/api/v2/seed_clips_index'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: cvToken, reroll: 0, play_ranges: cvPlayRanges || [] }),
    }).then(function (r) { return r.json(); }).then(function (data) {
        if (data && data.moments) {
            s.moments = data.moments; s.nMoments = data.moments.length;
            while (s.slotReroll.length < s.nMoments) s.slotReroll.push(0);
        }
        if (then) then();
    }).catch(function () { if (then) then(); });
}

function __v2SeedRenderClipNav() {
    const el = document.getElementById('cv-seed-clipnav');
    const s = __v2Seed;
    if (!el || !s) return;
    el.innerHTML = '';
    for (let i = 0; i < s.nMoments; i++) {
        const dot = document.createElement('button');
        dot.type = 'button';
        dot.className = 'cv-seed-clip-dot' + (i === s.index ? ' active' : '') +
            (s.selections[(s.slotReroll[i] || 0) + '_' + i] ? ' picked' : '');
        dot.title = 'Clip ' + (i + 1);
        dot.onclick = (function (idx) { return function () { __v2SeedLoadClip(idx); }; })(i);
        el.appendChild(dot);
    }
}

function __v2SeedSetLoading(on, msg) {
    const el = document.getElementById('cv-seed-loading');
    if (!el) return;
    __pfNetInto(el);                       // drifting net behind the message
    // Update only the message node — replacing the whole innerHTML each tick
    // would rebuild (and restart) the net animation every second.
    let box = el.querySelector('.cv-seed-loading-msg');
    if (!box) {
        box = document.createElement('div');
        box.className = 'cv-seed-loading-msg';
        el.appendChild(box);
    }
    if (msg != null) box.innerHTML = msg;
    el.classList.toggle('hidden', !on);
    if (!on) {
        __v2SeedStopLoadTicker();
        // First time a seed clip finishes building this session, the screen is
        // now tappable — alert the user (once) in case they stepped away during
        // the one-time model prep.
        if (!__pfSeedReadyAlerted) { __pfSeedReadyAlerted = true; pfAlertInputNeeded(); }
    }
}

// While a clip builds, show a live timer. The very first build of the session
// also warns that a one-time model preparation makes it take a bit longer.
function __v2SeedStartLoadTicker(index) {
    __v2SeedStopLoadTicker();
    const first = !__v2SeedBuilt;
    const started = Date.now();
    function render() {
        const secs = Math.floor((Date.now() - started) / 1000);
        let html = 'Loading clip ' + (index + 1) + '… <span style="opacity:.7">' +
            secs + 's</span>';
        if (first) {
            html += '<br><span style="opacity:.75;font-size:.82em">' +
                'First clip also prepares the detection model (a one-time step, ' +
                'up to a couple of minutes on a CPU). Later clips are quick.' +
                '</span>';
        }
        __v2SeedSetLoading(true, html);
    }
    render();
    __v2SeedLoadTimer = setInterval(render, 1000);
}

function __v2SeedStopLoadTicker() {
    if (__v2SeedLoadTimer) { clearInterval(__v2SeedLoadTimer); __v2SeedLoadTimer = null; }
}

function __v2SeedLoadClip(index) {
    const s = __v2Seed;
    if (!s) return;
    s.index = index;
    __v2SeedUpdateFinishBtn();       // reflects selections across all clips
    __v2SeedRenderClipNav();
    __v2SeedStopAnim();
    document.getElementById('cv-seed-nodes').innerHTML = '';

    const key = (s.slotReroll[index] || 0) + '_' + index;
    const hint = document.getElementById('cv-seed-hint');
    const cached = s.cache[key];
    if (cached) { __v2SeedShowClip(cached, key); __v2SeedPrefetchNext(); return; }

    __v2SeedStartLoadTicker(index);
    if (hint) {
        hint.innerText = __v2SeedBuilt
            ? 'Tracking players in this clip…'
            : 'Getting your first clip ready — this one takes a little longer.';
    }
    fetch(__v2SeedApi('/api/v2/seed_clip'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            token: cvToken, index: index, reroll: (s.slotReroll[index] || 0),
            my_team_hex: (cvMyTeam && cvMyTeam.hex) || null,
            avoid: __v2SeedAvoidFor(index),
            play_ranges: cvPlayRanges || [],
        }),
    }).then(function (r) { return r.json(); }).then(function (data) {
        if (!data || data.error || (!data.video_url && !data.clip_url)) {
            throw new Error((data && data.error) || 'clip failed');
        }
        __v2SeedBuilt = true;
        s.cache[key] = data;
        if (typeof data.t_center === 'number') s.chosen[index] = data.t_center;
        __v2SeedRecordMoment(index, data.t_center);   // remember it so shuffle won't repeat
        // Only render if the user hasn't navigated away while we loaded.
        if ((s.slotReroll[s.index] || 0) + '_' + s.index === key) { __v2SeedShowClip(data, key); }
        __v2SeedPrefetchNext();
    }).catch(function () {
        __v2SeedSetLoading(false);
        if (hint) hint.innerText = 'Could not build this clip. Try another, or shuffle.';
    });
}

function __v2SeedPrefetchNext() {
    const s = __v2Seed;
    if (!s) return;
    const nxt = s.index + 1;
    if (nxt >= s.nMoments) return;
    const key = (s.slotReroll[nxt] || 0) + '_' + nxt;
    if (s.cache[key]) return;
    fetch(__v2SeedApi('/api/v2/seed_clip'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            token: cvToken, index: nxt, reroll: (s.slotReroll[nxt] || 0),
            my_team_hex: (cvMyTeam && cvMyTeam.hex) || null,
            avoid: __v2SeedAvoidFor(nxt),
            play_ranges: cvPlayRanges || [],
        }),
    }).then(function (r) { return r.json(); }).then(function (data) {
        if (data && (data.video_url || data.clip_url)) {
            __v2SeedBuilt = true;
            s.cache[(s.slotReroll[nxt] || 0) + '_' + nxt] = data;
            if (typeof data.t_center === 'number') s.chosen[nxt] = data.t_center;
            __v2SeedRecordMoment(nxt, data.t_center);
        }
    }).catch(function () {});
}

function __v2SeedShowClip(clip, key) {
    const s = __v2Seed;
    s.clip = clip;
    if (clip && typeof clip.t_center === 'number') s.chosen[s.index] = clip.t_center;
    const vid = document.getElementById('cv-seed-video');
    const hint = document.getElementById('cv-seed-hint');
    __v2SeedSetLoading(false);
    __v2SeedRenderClipNav();
    const n = (clip.tracklets || []).length;
    const already = s.selections[key];
    if (hint) {
        if (!n) hint.innerText = 'No players tracked in this clip — try another or shuffle.';
        else if (already) hint.innerText = 'Marked in this clip. Check the other clips too — each one you mark yourself in improves accuracy.';
        else hint.innerText = 'Tap the marker on you (or skip if you\'re not here). Mark yourself in every clip you appear in.';
    }
    // One group per tracked player: a pin near the top + a thin leader line down
    // to the player, so the marker never blocks the view of the player.
    const tracks = clip.tracklets || [];
    // Team filter: split the clip's players into two kit colours and hide the
    // team you didn't pick. If that would hide *everyone* (bad colour read),
    // fall back to showing all.
    let hidden = __v2SeedTeamHiddenFlags(tracks);
    if (tracks.length && hidden.every(function (h) { return h; })) {
        hidden = tracks.map(function () { return false; });
    }
    s.teamFiltered = hidden.some(function (h) { return h; });
    const nodes = document.getElementById('cv-seed-nodes');
    nodes.innerHTML = '';
    s.nodeEls = [];
    tracks.forEach(function (tr, ti) {
        const group = document.createElement('div');
        group.className = 'cv-seed-track';
        group.dataset.trackId = String(tr.id);
        const leader = document.createElement('div');
        leader.className = 'cv-seed-leader';
        leader.style.display = 'none';
        const pin = document.createElement('div');
        pin.className = 'cv-seed-node';
        pin.style.display = 'none';
        pin.title = 'This is me';
        pin.onclick = function (e) { e.stopPropagation(); __v2SeedPickNode(tr, key); };
        group.appendChild(leader);
        group.appendChild(pin);
        nodes.appendChild(group);
        s.nodeEls.push({ group: group, pin: pin, leader: leader, hidden: hidden[ti] });
    });
    __v2SeedUpdateShowAll();
    // Re-highlight the marker you picked earlier in this clip, if any, and
    // restore its switch flag; otherwise clear the flag row for a fresh clip.
    if (already) {
        s.nodeEls.forEach(function (e) {
            e.group.classList.toggle('selected',
                e.group.dataset.trackId === String(already.trackId));
        });
        var picked = tracks.filter(function (t) {
            return String(t.id) === String(already.trackId);
        })[0];
        __v2SeedSetFlag(true, !!(picked && __v2SeedTrackSwitched(picked)));
    } else {
        __v2SeedSetFlag(false, false);
    }
    __v2SeedUpdateFinishBtn();
    __v2SeedShowPlayPause(false);   // starts playing → hide indicator
    __v2SeedAttachVideo(vid, clip);
    __v2SeedStartAnim();
}

// Play a seed moment from the uploaded video (no separate encoded clip).
function __v2SeedAttachVideo(vid, clip) {
    if (!vid || !clip) return;
    var url = cvVideoURL || cvVideoUrlForToken(cvToken);
    var start = clip.start_sec || 0;
    var end = start + (clip.duration_sec || 3);
    function seekStart() {
        try { vid.currentTime = start; } catch (e) {}
        var p = vid.play();
        if (p && p.catch) p.catch(function () {});
    }
    if (vid.getAttribute('data-seed-src') !== url) {
        vid.setAttribute('data-seed-src', url);
        vid.src = url;
        vid.load();
        vid.addEventListener('loadeddata', function ol() {
            vid.removeEventListener('loadeddata', ol);
            seekStart();
        });
    } else if (vid.readyState >= 1) {
        seekStart();
    }
    vid.ontimeupdate = function () {
        if (vid.currentTime >= end || vid.currentTime < start - 0.05) {
            try { vid.currentTime = start; } catch (e) {}
        }
    };
}

// Clip-local playback time (tracklet points are 0 … duration_sec).
function __v2SeedLocalT(vid, clip) {
    return Math.max(0, (vid && vid.currentTime || 0) - ((clip && clip.start_sec) || 0));
}

// Node anchor: markers sit this far down from the top of the clip.
const __V2_SEED_TOP_PCT = 6;

// Interpolate a tracklet's normalized position at clip-local time t.
function __v2SeedPosAt(points, t) {
    if (!points || !points.length) return null;
    if (t <= points[0].t) return points[0].t - t < 0.4 ? points[0] : null;
    const last = points[points.length - 1];
    if (t >= last.t) return t - last.t < 0.4 ? last : null;
    for (let i = 0; i < points.length - 1; i++) {
        const a = points[i], b = points[i + 1];
        if (t >= a.t && t <= b.t) {
            const f = (b.t - a.t) > 1e-6 ? (t - a.t) / (b.t - a.t) : 0;
            return { nx: a.nx + (b.nx - a.nx) * f, ny: a.ny + (b.ny - a.ny) * f };
        }
    }
    return null;
}

function __v2SeedStartAnim() {
    __v2SeedStopAnim();
    const vid = document.getElementById('cv-seed-video');
    const TOP = __V2_SEED_TOP_PCT;
    function frame() {
        const s = __v2Seed;
        if (!s || !s.clip) return;
        // currentTime is constant while paused, so the markers freeze in place.
        const t = __v2SeedLocalT(vid, s.clip);
        const els = s.nodeEls || [];
        const tracks = s.clip.tracklets || [];
        for (let i = 0; i < tracks.length && i < els.length; i++) {
            const e = els[i];
            const pos = __v2SeedPosAt(tracks[i].points, t);
            if (!pos || (e.hidden && !s.showAll)) {
                e.pin.style.display = 'none'; e.leader.style.display = 'none'; continue;
            }
            const xPct = pos.nx * 100, yPct = pos.ny * 100;
            e.pin.style.display = 'block';
            e.pin.style.left = xPct + '%';
            e.pin.style.top = TOP + '%';
            const h = yPct - TOP;                       // pin(top) → player line
            e.leader.style.display = h > 0.5 ? 'block' : 'none';
            e.leader.style.left = xPct + '%';
            e.leader.style.top = TOP + '%';
            e.leader.style.height = Math.max(0, h) + '%';
        }
        __v2SeedRAF = requestAnimationFrame(frame);
    }
    __v2SeedRAF = requestAnimationFrame(frame);
}

function __v2SeedTogglePlay() {
    const s = __v2Seed;
    if (!s || !s.clip) return;              // nothing to toggle while loading
    const vid = document.getElementById('cv-seed-video');
    if (!vid) return;
    if (vid.paused) {
        const p = vid.play();
        if (p && p.catch) p.catch(function () {});
        __v2SeedShowPlayPause(false);
    } else {
        vid.pause();
        __v2SeedShowPlayPause(true);
    }
}

// Show ▶ persistently while paused; flash the pause glyph briefly when resuming.
let __v2SeedPPTimer = null;
function __v2SeedShowPlayPause(paused) {
    const ind = document.getElementById('cv-seed-playpause');
    if (!ind) return;
    if (__v2SeedPPTimer) { clearTimeout(__v2SeedPPTimer); __v2SeedPPTimer = null; }
    if (paused) {
        ind.textContent = '▶';
        ind.classList.remove('hidden');
    } else {
        ind.textContent = '❚❚';
        ind.classList.remove('hidden');
        __v2SeedPPTimer = setTimeout(function () { ind.classList.add('hidden'); }, 500);
    }
}

function __v2SeedStopAnim() {
    if (__v2SeedRAF) { cancelAnimationFrame(__v2SeedRAF); __v2SeedRAF = null; }
}

// Build the appearance sample from a short window around the moment you tapped
// (you confirmed it was you then), not the whole clip — so a tag that switches a
// second later never poisons your gallery.
const __V2_SEED_TAP_WINDOW = 0.75;   // seconds each side of the tap
function __v2SeedWindowTaps(tr, clip, tapT) {
    const start = clip.start_sec || 0;
    const pts = tr.points || [];
    if (!pts.length) return tr.taps || [];
    let win = pts.filter(function (p) {
        return Math.abs(p.t - tapT) <= __V2_SEED_TAP_WINDOW;
    });
    if (win.length < 3) {                       // sparse here → nearest few points
        win = pts.slice().sort(function (a, b) {
            return Math.abs(a.t - tapT) - Math.abs(b.t - tapT);
        }).slice(0, Math.min(6, pts.length));
    }
    if (win.length > 8) {                        // cap, spread across the window
        const step = win.length / 8, out = [];
        for (let i = 0; i < 8; i++) out.push(win[Math.floor(i * step)]);
        win = out;
    }
    return win.map(function (p) {
        return { t_sec: Math.round((start + p.t) * 1000) / 1000, nx: p.nx, ny: p.ny };
    });
}

// A tag likely switched players if the tracked colour flips across the clip:
// compare the median colour of the early points to the late points.
function __v2SeedTrackSwitched(tr) {
    const cols = (tr.points || []).map(function (p) { return p.c; }).filter(Boolean);
    if (cols.length < 4) return false;
    function med(arr) {
        const m = [0, 1, 2].map(function (k) {
            const v = arr.map(function (c) { return c[k]; }).sort(function (a, b) { return a - b; });
            return v[Math.floor(v.length / 2)];
        });
        return m;
    }
    const n = cols.length, third = Math.max(1, Math.floor(n * 0.4));
    const early = med(cols.slice(0, third)), late = med(cols.slice(n - third));
    const d = __hsvDist(early, late);
    return d != null && d > 55;                  // clear colour flip → probable switch
}

function __v2SeedPickNode(tr, key) {
    const s = __v2Seed;
    if (!s) return;
    const vid = document.getElementById('cv-seed-video');
    const tapT = __v2SeedLocalT(vid, s.clip);
    const cur = s.selections[key];
    const deselect = cur && String(cur.trackId) === String(tr.id);  // tap again → clear
    if (deselect) {
        delete s.selections[key];
    } else {
        s.selections[key] = {
            trackId: tr.id,
            taps: __v2SeedWindowTaps(tr, s.clip, tapT),   // tap-anchored sample
        };
    }
    (s.nodeEls || []).forEach(function (e) {
        e.group.classList.toggle('selected',
            !deselect && e.group.dataset.trackId === String(tr.id));
    });
    // Auto-flag a likely tag switch on the player you picked.
    const switched = !deselect && __v2SeedTrackSwitched(tr);
    __v2SeedSetFlag(!deselect, switched);
    const hint = document.getElementById('cv-seed-hint');
    if (hint) {
        const nsel = __v2SeedSelCount();
        hint.innerText = deselect
            ? 'Cleared this clip. Tap yourself, or move to another clip.'
            : 'Got you in this clip (' + nsel + (nsel === 1 ? ' clip' : ' clips') +
              ' marked). Check the other clips too, then find your touches.';
    }
    __v2SeedUpdateFinishBtn();
    __v2SeedRenderClipNav();
}

// Show/hide the "this clip is wrong" row; highlight it when a switch is detected.
function __v2SeedSetFlag(show, switched) {
    const row = document.getElementById('cv-seed-flag');
    const warn = document.getElementById('cv-seed-flag-warn');
    if (!row) return;
    row.classList.toggle('hidden', !show);
    if (warn) {
        warn.textContent = switched
            ? '⚠ This tag looks like it jumped to another player partway through.'
            : '';
        warn.classList.toggle('hidden', !switched);
    }
    row.classList.toggle('flagged', !!switched);
}

// Manual override: this clip's tracking is unreliable → drop it and move on.
function __v2SeedRejectClip() {
    const s = __v2Seed;
    if (!s) return;
    const key = (s.slotReroll[s.index] || 0) + '_' + s.index;
    delete s.selections[key];
    (s.nodeEls || []).forEach(function (e) { e.group.classList.remove('selected'); });
    __v2SeedSetFlag(false, false);
    __v2SeedUpdateFinishBtn();
    __v2SeedRenderClipNav();
    __v2SeedNextClip();
}

// Shuffle THIS clip only — reshuffles the current slot to a fresh moment,
// leaving the other clips (and anything you've already marked in them) intact.
function __v2SeedReroll() {
    const s = __v2Seed;
    if (!s) return;
    const i = s.index;
    // Drop this slot's current mark — it's about to be a different moment — but
    // keep every other clip's selection so already-good clips still count.
    delete s.selections[(s.slotReroll[i] || 0) + '_' + i];
    delete s.chosen[i];    // this slot's old moment no longer constrains others
    s.slotReroll[i] = (s.slotReroll[i] || 0) + 1;
    __v2SeedUpdateFinishBtn();
    __v2SeedLoadClip(i);   // rebuild only this slot, at its new reroll
}

function __v2SeedNextClip() {
    const s = __v2Seed;
    if (!s) return;
    __v2SeedLoadClip((s.index + 1) % Math.max(1, s.nMoments));
}

function __v2SeedFinish() {
    const s = __v2Seed;
    const taps = __v2SeedCombinedTaps();
    if (taps.length === 0) {
        const ok = window.confirm(
            'You haven\'t marked yourself in any clip yet.\n\n' +
            'Analysis still runs, but without an appearance model it can\'t rank ' +
            'your touches as well — you\'ll just review more clips. Continue anyway?');
        if (!ok) return;
    }
    // Marking 2-3 clips is more reliable than 1 (the other clips cover for a bad
    // tag), and clip-nav + "Not here — next clip" already make that path clear —
    // so we nudge via the inline hint text (see __v2SeedPickNode), not a blocking
    // dialog on the way out. The button itself always reflects how many you've
    // marked ("Find my touches · N clips"), so the choice is informed either way.
    __v2SeedStopAnim();
    var vid = document.getElementById('cv-seed-video');
    if (vid) { try { vid.pause(); } catch (e) {} }
    document.getElementById('cv-seed-screen').classList.add('hidden');
    startCvV2Analysis(taps);
}

document.addEventListener('DOMContentLoaded', function () {
    initProcessTracker();
    tryResumeCvSession();
    refreshMatchCatalogue();
    // Keep the waiting screens alive with the homepage's drifting net.
    __pfNetInto(document.getElementById('cv-processing-screen'));
    __pfNetInto(document.getElementById('cv-seed-loading'));
    const teamCancel = document.getElementById('cv-team-cancel');
    const procCancel = document.getElementById('cv-processing-cancel');
    const resumeBtn = document.getElementById('cv-resume-btn');
    const discardBtn = document.getElementById('cv-discard-btn');
    const catalogueRefresh = document.getElementById('catalogue-refresh-btn');
    if (resumeBtn) resumeBtn.addEventListener('click', resumeCvFromBanner);
    if (discardBtn) discardBtn.addEventListener('click', confirmDiscardRun);
    var discardDoneBtn = document.getElementById('cv-discard-done-btn');
    if (discardDoneBtn) discardDoneBtn.addEventListener('click', confirmDiscardRun);
    if (catalogueRefresh) catalogueRefresh.addEventListener('click', refreshMatchCatalogue);
    window.addEventListener('beforeunload', function () {
        if (cvJobId) saveMatchSession(cvJobId);
    });
    if (teamCancel) teamCancel.addEventListener('click', function () {
        document.getElementById('cv-team-screen').classList.add('hidden');
        hideProcessTracker();
        const setupScreen = document.getElementById('setup-screen');
        if (setupScreen) setupScreen.classList.remove('hidden');
    });
    if (procCancel) procCancel.addEventListener('click', cancelCvAnalysis);

    // v2 tracked-node seed controls
    var seedNext = document.getElementById('cv-seed-next');
    var seedSkip = document.getElementById('cv-seed-skip');
    var seedReroll = document.getElementById('cv-seed-reroll');
    var seedCancel = document.getElementById('cv-seed-cancel');
    if (seedNext) seedNext.addEventListener('click', function () { __v2SeedFinish(); });
    if (seedSkip) seedSkip.addEventListener('click', function () { __v2SeedNextClip(); });
    if (seedReroll) seedReroll.addEventListener('click', function () { __v2SeedReroll(); });
    var seedReject = document.getElementById('cv-seed-reject');
    if (seedReject) seedReject.addEventListener('click', function () { __v2SeedRejectClip(); });
    var seedStage = document.getElementById('cv-seed-stage');
    if (seedStage) seedStage.addEventListener('click', function () { __v2SeedTogglePlay(); });
    var seedShowAll = document.getElementById('cv-seed-showall');
    if (seedShowAll) seedShowAll.addEventListener('change', function () {
        __v2SeedToggleShowAll(this.checked);
    });
    if (seedCancel) seedCancel.addEventListener('click', function () {
        __v2SeedStopAnim();
        var sv = document.getElementById('cv-seed-video');
        if (sv) { try { sv.pause(); } catch (e) {} }
        document.getElementById('cv-seed-screen').classList.add('hidden');
        hideProcessTracker();
        var setupScreen = document.getElementById('setup-screen');
        if (setupScreen) setupScreen.classList.remove('hidden');
    });
});

// --- 4b-v2. RUN v2 ANALYSIS (token + seed taps -> montage review) ---
function startCvV2Analysis(taps) {
    const proc = document.getElementById('cv-processing-screen');
    proc.classList.remove('hidden');
    showProcessTracker();
    setCvProgress({ progress: 0, status: 'Starting single-player analysis…', stage: 'init' });

    if (!cvToken) { runBrowserDemo(); return; }

    const fd = new FormData();
    fd.append('token', cvToken);
    fd.append('seed_taps', JSON.stringify(taps || []));
    // Belt and braces: the server prefers its own stored playing_time.json, but
    // sending it here covers the case where /api/v2/playing_time never landed
    // (offline blip, server restart mid-flow).
    fd.append('play_ranges', JSON.stringify(cvPlayRanges || []));
    if (cvMyTeamId) fd.append('my_team', cvMyTeamId);
    // The OTHER detected kit colour (the team the user didn't pick) → lets the
    // pipeline's team gate positively identify and drop wrong-team touches.
    var oppTeam = (cvTeams || []).filter(function (t) {
        return t && t.id !== cvMyTeamId;
    })[0];
    if (oppTeam && oppTeam.hex) fd.append('opponent_hex', oppTeam.hex);
    // Full colour sets too: a kit that isn't one flat colour is matched on any
    // of its colours, so all of them have to reach the team gate.
    var oppHexes = __cvTeamHexes(oppTeam);
    if (oppHexes.length) fd.append('opponent_hexes', JSON.stringify(oppHexes));
    var mineHexes = __cvTeamHexes(cvMyTeam);
    if (mineHexes.length) fd.append('my_team_hexes', JSON.stringify(mineHexes));
    // Pitch calibration (optional). The server re-fits these clicks itself, so
    // what rides along is the marks plus our fit as a starting hint.
    if (cvPitchCalibration) {
        fd.append('calibration', JSON.stringify(cvPitchCalibration));
    }
    var meta = getSetupMetadataFields();
    fd.append('opponent', meta.opponent || '');
    fd.append('match_date', meta.match_date || '');
    fd.append('score_us', String(meta.score_us != null ? meta.score_us : 0));
    fd.append('score_them', String(meta.score_them != null ? meta.score_them : 0));
    fd.append('position', meta.position || '');

    fetch(cvApiUrl('/api/v2/process'), { method: 'POST', body: fd })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) throw new Error(data.error);
            cvJobId = data.job_id;
            cvSegmentsAreDemo = false;
            saveCvSession({ job_id: cvJobId, token: cvToken, my_team: cvMyTeamId, state: 'running', v2: true });
            captureSetupMetadataToSession();
            pollCvStatus();
        })
        .catch(function () { runBrowserDemo(); });
}

function enterV2Review(j) {
    hideProcessTracker();
    document.getElementById('cv-processing-screen').classList.add('hidden');
    pfAlertInputNeeded();   // analysis done — the "was this you?" review needs the user
    showV2Montage(j);
}

// --- 4d-v2. ME / NOT-ME MONTAGE REVIEW ---
let __v2Montage = null;
let __v2MontageTemplate = null;

function __v2FmtTime(sec) {
    sec = Math.max(0, Math.round(sec || 0));
    var m = Math.floor(sec / 60), s = sec % 60;
    return m + ':' + (s < 10 ? '0' : '') + s;
}

function showV2Montage(j) {
    var items = (j && j.montage) ? j.montage.slice() : [];
    var queue = items.filter(function (it) { return it.status === 'review'; });
    __v2Montage = {
        jobId: (j && j.job_id) || cvJobId, token: (j && j.token) || cvToken,
        all: items, queue: queue, idx: 0, decisions: {},
        duration: (j && j.duration_sec) || 0, playing: false,
        total0: queue.length, done: 0, lastAuto: null,   // adaptive-grouping progress
        autoAccept: items.filter(function (it) { return it.status === 'auto_accept'; }).length,
        autoHide: items.filter(function (it) { return it.status === 'auto_hide'; }).length,
        warnings: (j && j.warnings) || []
    };
    pfShowPipelineWarnings(__v2Montage.warnings);

    var screen = document.getElementById('cv-montage-screen');
    var card = screen.querySelector('.cv-montage-card');
    if (__v2MontageTemplate === null) __v2MontageTemplate = card.innerHTML;
    else card.innerHTML = __v2MontageTemplate;   // restore original layout for repeat runs
    screen.classList.remove('hidden');

    // Wire controls (re-created on restore).
    var meBtn = document.getElementById('cv-montage-me');
    var notMeBtn = document.getElementById('cv-montage-notme');
    var finishBtn = document.getElementById('cv-montage-finish');
    if (meBtn) meBtn.onclick = function () { __v2MontageDecide('me'); };
    if (notMeBtn) notMeBtn.onclick = function () { __v2MontageDecide('not_me'); };
    if (finishBtn) finishBtn.onclick = function () { __v2MontageFinalize(); };
    var undoBtn = document.getElementById('cv-montage-undo-btn');
    if (undoBtn) undoBtn.onclick = function () { __v2MontageUndo(); };

    // zoom / pan controls (re-created on restore)
    var zin = document.getElementById('cv-montage-zoomin');
    var zout = document.getElementById('cv-montage-zoomout');
    var zreset = document.getElementById('cv-montage-zoomreset');
    if (zin) zin.onclick = function () { __v2MontageZoomStep(1.4); };
    if (zout) zout.onclick = function () { __v2MontageZoomStep(1 / 1.4); };
    if (zreset) zreset.onclick = function () { __v2MontageResetView(); };
    __v2MontageWireCanvas();

    var vid = document.getElementById('cv-montage-video');
    vid.removeAttribute('data-src'); vid.removeAttribute('src');   // __v2MontageLoadSource sets it

    if (queue.length === 0) { __v2MontageFinalize(); return; }
    __v2MontageShow();
}

// Wheel-zoom and drag-to-pan on the review canvas.
function __v2MontageWireCanvas() {
    var canvas = document.getElementById('cv-montage-canvas');
    if (!canvas) return;
    canvas.onwheel = function (e) {
        e.preventDefault();
        var r = canvas.getBoundingClientRect();
        var cx = (e.clientX - r.left) / r.width, cy = (e.clientY - r.top) / r.height;
        var m = __v2Montage; if (!m) return;
        __v2MontageZoomAt((m.zoom || 1) * (e.deltaY < 0 ? 1.15 : 1 / 1.15), cx, cy);
    };
    var drag = null;
    canvas.onpointerdown = function (e) {
        var m = __v2Montage; if (!m || (m.zoom || 1) <= 1.001) return;
        drag = { x: e.clientX, y: e.clientY, panX: m.panX, panY: m.panY };
        canvas.classList.add('grabbing');
        try { canvas.setPointerCapture(e.pointerId); } catch (err) {}
    };
    canvas.onpointermove = function (e) {
        if (!drag) return;
        var m = __v2Montage; if (!m) return;
        var r = canvas.getBoundingClientRect();
        m.panX = drag.panX - (e.clientX - drag.x) / r.width / (m.zoom || 1);
        m.panY = drag.panY - (e.clientY - drag.y) / r.height / (m.zoom || 1);
        __v2MontageClampPan();
    };
    canvas.onpointerup = canvas.onpointercancel = function (e) {
        drag = null; canvas.classList.remove('grabbing');
        try { canvas.releasePointerCapture(e.pointerId); } catch (err) {}
    };
}

function __v2MontageRenderProgress() {
    var m = __v2Montage; if (!m) return;
    var el = document.getElementById('cv-montage-progress');
    if (!el) return;
    var total = Math.max(1, m.total0 || 0);
    var left = m.queue.length;
    el.innerHTML = '<div class="cv-montage-bar"><div class="cv-montage-bar-fill" style="width:' +
        Math.round(100 * m.done / total) + '%"></div></div>' +
        '<span class="cv-montage-count">' + left + ' touch' + (left === 1 ? '' : 'es') +
        ' left to review</span>';
}

// Interpolate a review tracklet's normalized centre at absolute video time t.
// Outside the tracked span → null (hide the box; do not hold last position).
function __v2MontagePosAt(points, t) {
    if (!points || !points.length) return null;
    var first = points[0], last = points[points.length - 1];
    var pad = 0.04;  // ~1 frame at 25fps
    if (t < first.t_sec - pad || t > last.t_sec + pad) return null;
    if (t <= first.t_sec) return first;
    if (t >= last.t_sec) return last;
    for (var i = 0; i < points.length - 1; i++) {
        var a = points[i], b = points[i + 1];
        if (t >= a.t_sec && t <= b.t_sec) {
            // Gap in the track (lost lock, then later points) → hide in between
            if ((b.t_sec - a.t_sec) > 0.2) return null;
            var f = (b.t_sec - a.t_sec) > 1e-6 ? (t - a.t_sec) / (b.t_sec - a.t_sec) : 0;
            return { nx: a.nx + (b.nx - a.nx) * f, ny: a.ny + (b.ny - a.ny) * f };
        }
    }
    return null;
}

// Fetch (and cache on the item) the tracklet that makes the ring follow the
// reviewed player. Falls back silently to player_bbox / contact marker.
function __v2MontageFetchTrack(it) {
    var m = __v2Montage;
    if (!m || !it || it.__track !== undefined) return;
    it.__track = null;   // "requested" — avoid duplicate fetches
    fetch(cvApiUrl('/api/v2/review_track/' + m.jobId + '/' + it.rank))
        .then(function (r) { return r.json(); })
        .then(function (d) { it.__track = (d && d.points) ? d.points : []; })
        .catch(function () { it.__track = []; });
}

// Pipeline boxes live in target_width=640 space; map to the video's native px.
function __v2MontageNativeScale(nW) {
    return nW / 640;
}

// Body box for the attributed player, in native-frame pixels.
// Only when identity_linked: box the continuity-linked player. Hide when the
// track is lost or the contact is uncertain (user sorts those clips).
function __v2MontagePlayerBox(it, absT, nW, nH) {
    if (it.identity_linked === false) return null;
    var sc = __v2MontageNativeScale(nW);
    var pb = it.player_bbox;
    var nearTouch = (typeof it.t_sec === 'number') && Math.abs(absT - it.t_sec) <= 0.2;
    if (it.__track && it.__track.length) {
        var pos = __v2MontagePosAt(it.__track, absT);
        if (pos) {
            var p0 = it.__track[0];
            return {
                cx: pos.nx * nW,
                cy: pos.ny * nH,
                w: ((p0 && p0.nw) ? p0.nw : 0.055) * nW,
                h: ((p0 && p0.nh) ? p0.nh : 0.14) * nH
            };
        }
        // Track exists but lost lock at this time → hide (do not freeze last box)
        if (!nearTouch) return null;
    }
    if (pb && pb.length === 4 && nearTouch) {
        return {
            cx: ((pb[0] + pb[2]) / 2) * sc,
            cy: ((pb[1] + pb[3]) / 2) * sc,
            w: (pb[2] - pb[0]) * sc,
            h: (pb[3] - pb[1]) * sc
        };
    }
    if (pb && pb.length === 4 && !(it.__track && it.__track.length)) {
        return {
            cx: ((pb[0] + pb[2]) / 2) * sc,
            cy: ((pb[1] + pb[3]) / 2) * sc,
            w: (pb[2] - pb[0]) * sc,
            h: (pb[3] - pb[1]) * sc
        };
    }
    return null;
}

// Ball / touch point from the montage crop (always ball-centered).
function __v2MontageBallPoint(it, nW) {
    if (!it.crop || it.crop.length !== 4) return null;
    var sc = __v2MontageNativeScale(nW);
    return {
        cx: ((it.crop[0] + it.crop[2]) / 2) * sc,
        cy: ((it.crop[1] + it.crop[3]) / 2) * sc
    };
}

function __v2MontageShow() {
    var m = __v2Montage; if (!m) return;
    var it = m.queue[0];
    if (!it) { __v2MontageFinalize(); return; }
    m.curItem = it;
    m.zoom = 1; m.panX = 0.5; m.panY = 0.5;        // reset view for each clip
    m.timeOffset = 0;
    __v2MontageUpdateZoomUI();
    __v2MontageRenderProgress();
    var meta = document.getElementById('cv-montage-meta');
    if (meta) {
        var crowdBadge = it.crowded
            ? '<span class="cv-montage-crowded" title="' + (it.n_nearby_players || 0) +
              ' players around the ball — we can\'t tell who touched it">Crowded</span>'
            : '';
        // Tell the user WHY there is (or isn't) a box: linked = we followed
        // you here from a confirmed sighting; uncertain = look at the ball.
        var idBadge = (it.identity_linked === false)
            ? '<span class="cv-montage-unlinked" title="We lost track of you before this touch, so no box is drawn — judge from the play">Uncertain — no box</span>'
            : '<span class="cv-montage-linked" title="Tracked from a confirmed sighting of you — the box follows that player">Tracked</span>';
        meta.innerHTML =
            '<span class="cv-montage-time">' + __v2FmtTime(it.t_sec) + '</span>' +
            '<span class="cv-montage-kinds">' + (it.kinds || []).join(' · ') + '</span>' +
            crowdBadge + idBadge +
            '<span class="cv-montage-conf">match ' + Math.round((it.confidence || 0) * 100) + '%</span>';
    }
    __v2MontageFetchTrack(it);                     // ring-follow track for this touch
    if (m.queue[1]) __v2MontageFetchTrack(m.queue[1]);  // prefetch next
    m.playing = true;
    __v2MontageLoadSource();
    __v2MontageStartDraw();
}

// Point the video at the source clip window without disturbing zoom/pan.
function __v2MontageLoadSource() {
    var m = __v2Montage; if (!m || !m.curItem) return;
    var it = m.curItem;
    var vid = document.getElementById('cv-montage-video');
    var srcUrl = cvApiUrl('/api/video/' + m.token);
    var start = it.clip_start_sec, end = it.clip_end_sec;
    m.timeOffset = 0;
    m.playStart = start; m.playEnd = end;
    function seekStart() {
        try { vid.currentTime = start; } catch (e) {}
        var p = vid.play(); if (p && p.catch) p.catch(function () {});
    }
    if (vid.getAttribute('data-src') !== srcUrl) {
        vid.setAttribute('data-src', srcUrl);
        vid.src = srcUrl; vid.load();
        vid.addEventListener('loadeddata', function ol() {
            vid.removeEventListener('loadeddata', ol); seekStart();
        });
    } else if (vid.readyState >= 1) {
        seekStart();
    }
    vid.ontimeupdate = function () {
        if (vid.currentTime >= m.playEnd || vid.currentTime < m.playStart - 0.2) {
            try { vid.currentTime = m.playStart; } catch (e) {}
        }
    };
}

// One draw loop for the whole review; reads the live zoom/pan + current clip.
function __v2MontageStartDraw() {
    var m = __v2Montage; if (!m || m.__drawing) return;
    m.__drawing = true;
    var vid = document.getElementById('cv-montage-video');
    var canvas = document.getElementById('cv-montage-canvas');
    var ctx = canvas.getContext('2d');
    function draw() {
        var mm = __v2Montage;
        if (!mm || mm !== m || !mm.playing) { m.__drawing = false; return; }
        var it = mm.curItem;
        var nW = vid.videoWidth, nH = vid.videoHeight;
        if (it && nW && nH) {
            var bufW = Math.min(nW, 960);                    // sharper buffer for zoom
            var bufH = Math.round(bufW * nH / nW);
            if (canvas.width !== bufW || canvas.height !== bufH) {
                canvas.width = bufW; canvas.height = bufH;
            }
            // Digital zoom/pan: draw a sub-region of the frame to fill the canvas.
            var zoom = mm.zoom || 1;
            var sw = nW / zoom, sh = nH / zoom;
            var sx = Math.max(0, Math.min(nW - sw, (mm.panX || 0.5) * nW - sw / 2));
            var sy = Math.max(0, Math.min(nH - sh, (mm.panY || 0.5) * nH - sh / 2));
            ctx.drawImage(vid, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);

            // Green box = continuity-linked "you" only. Yellow dot = ball.
            // Hide the box when identity is uncertain or the track is lost.
            var absT = (vid.currentTime || 0) + (mm.timeOffset || 0);
            var box = __v2MontagePlayerBox(it, absT, nW, nH);
            if (box && box.w > 0 && box.h > 0) {
                var bboxW = box.w * 1.06, bboxH = box.h * 1.08;
                var rx = (box.cx - bboxW / 2 - sx) / sw * canvas.width;
                var ry = (box.cy - bboxH / 2 - sy) / sh * canvas.height;
                var rw = bboxW / sw * canvas.width;
                var rh = bboxH / sh * canvas.height;
                ctx.lineWidth = Math.max(2, canvas.width * 0.004);
                ctx.strokeStyle = 'rgba(48,255,143,0.95)';
                ctx.shadowColor = 'rgba(0,0,0,0.65)';
                ctx.shadowBlur = 4;
                ctx.strokeRect(rx, ry, rw, rh);
                ctx.shadowBlur = 0;
            }
            var ball = __v2MontageBallPoint(it, nW);
            if (ball) {
                var bcx = (ball.cx - sx) / sw * canvas.width;
                var bcy = (ball.cy - sy) / sh * canvas.height;
                var br = Math.max(3, canvas.width * 0.006);
                ctx.beginPath();
                ctx.arc(bcx, bcy, br, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255, 220, 60, 0.95)';
                ctx.strokeStyle = 'rgba(0,0,0,0.55)';
                ctx.lineWidth = Math.max(1, canvas.width * 0.002);
                ctx.fill();
                ctx.stroke();
            }
        }
        requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
}

// --- zoom / pan controls ---
function __v2MontageUpdateZoomUI() {
    var m = __v2Montage; if (!m) return;
    var lvl = document.getElementById('cv-montage-zoomlevel');
    if (lvl) lvl.textContent = (m.zoom || 1).toFixed(1) + '×';
    var out = document.getElementById('cv-montage-zoomout');
    if (out) out.disabled = (m.zoom || 1) <= 1.001;
}

function __v2MontageClampPan() {
    var m = __v2Montage; var half = 0.5 / (m.zoom || 1);
    m.panX = Math.max(half, Math.min(1 - half, m.panX));
    m.panY = Math.max(half, Math.min(1 - half, m.panY));
}

// Zoom toward a point given as fractions of the canvas [0,1], keeping that point
// fixed under the cursor.
function __v2MontageZoomAt(newZoom, cxFrac, cyFrac) {
    var m = __v2Montage; if (!m) return;
    newZoom = Math.max(1, Math.min(6, newZoom));
    var zOld = m.zoom || 1;
    var swOld = 1 / zOld;
    var sxOld = Math.max(0, Math.min(1 - swOld, (m.panX || 0.5) - swOld / 2));
    var syOld = Math.max(0, Math.min(1 - swOld, (m.panY || 0.5) - swOld / 2));
    var fXn = sxOld + cxFrac * swOld, fYn = syOld + cyFrac * swOld;
    var swNew = 1 / newZoom;
    m.zoom = newZoom;
    if (newZoom <= 1.001) { m.panX = 0.5; m.panY = 0.5; }
    else {
        m.panX = (fXn - cxFrac * swNew) + swNew / 2;
        m.panY = (fYn - cyFrac * swNew) + swNew / 2;
        __v2MontageClampPan();
    }
    __v2MontageUpdateZoomUI();
}

function __v2MontageZoomStep(factor) {
    var m = __v2Montage; if (!m) return;
    __v2MontageZoomAt((m.zoom || 1) * factor, 0.5, 0.5);
}

function __v2MontageResetView() {
    var m = __v2Montage; if (!m) return;
    m.zoom = 1; m.panX = 0.5; m.panY = 0.5;
    __v2MontageUpdateZoomUI();
}

// Does deciding `it` as `dec` also settle queued touch `q`?
//  • colour: "Not me" on the OTHER team clears every other same-kit opponent.
//  • appearance: within your kit, a look-alike group settles both ways (soft).
//  • crowded touches (corner kicks / scrambles) never auto-settle either way —
//    the kit/look-alike grouping behind propagation comes from an attribution
//    we don't trust in a pack, so the user judges each one.
function __v2MontageMatches(it, q, dec) {
    if (it.crowded || q.crowded) return null;
    if (dec === 'not_me' && it.is_other_team && q.is_other_team &&
        q.kit_group >= 0 && q.kit_group === it.kit_group) {
        return 'hard';
    }
    if (!it.is_other_team && !q.is_other_team &&
        it.appearance_group >= 0 && q.appearance_group === it.appearance_group) {
        return 'soft';
    }
    return null;
}

function __v2MontageDecide(dec) {
    var m = __v2Montage; if (!m) return;
    var it = m.queue[0];
    if (!it) return;
    m.decisions[it.rank] = dec;

    // Pull auto-settled touches out of the rest of the queue.
    var kept = [], removed = [], softCount = 0;
    for (var i = 1; i < m.queue.length; i++) {
        var q = m.queue[i];
        var kind = __v2MontageMatches(it, q, dec);
        if (kind) {
            m.decisions[q.rank] = dec;
            removed.push(q);
            if (kind === 'soft') softCount++;
        } else {
            kept.push(q);
        }
    }
    m.queue = kept;
    m.done += 1 + removed.length;
    m.lastAuto = removed.length ? { items: removed, dec: dec } : null;
    if (removed.length) __v2MontageShowUndo(removed.length, dec, softCount);
    else __v2MontageHideUndo();

    if (m.queue.length === 0) { __v2MontageFinalize(); return; }
    __v2MontageShow();
}

function __v2MontageShowUndo(n, dec, softCount) {
    var el = document.getElementById('cv-montage-undo');
    var txt = document.getElementById('cv-montage-undo-text');
    if (!el || !txt) return;
    var verb = dec === 'me' ? 'accepted' : 'cleared';
    var kind = (softCount >= n) ? 'look-alike' : (softCount > 0 ? 'related' : 'same-kit');
    txt.textContent = 'Also ' + verb + ' ' + n + ' ' + kind + ' touch' +
        (n === 1 ? '' : 'es') + '.';
    el.classList.remove('hidden');
}

function __v2MontageHideUndo() {
    var el = document.getElementById('cv-montage-undo');
    if (el) el.classList.add('hidden');
}

function __v2MontageUndo() {
    var m = __v2Montage; if (!m || !m.lastAuto) return;
    var a = m.lastAuto;
    a.items.forEach(function (q) { delete m.decisions[q.rank]; });
    m.queue = a.items.concat(m.queue);   // reinsert the auto-settled touches at the front
    m.done -= a.items.length;
    m.lastAuto = null;
    __v2MontageHideUndo();
    __v2MontageShow();
}

function __v2MontageStopVideo() {
    if (__v2Montage) __v2Montage.playing = false;
    var vid = document.getElementById('cv-montage-video');
    if (vid) { try { vid.pause(); } catch (e) {} vid.ontimeupdate = null; }
}

function __v2MontageFinalize() {
    var m = __v2Montage; if (!m) return;
    __v2MontageStopVideo();
    var meta = document.getElementById('cv-montage-meta');
    if (meta) meta.innerHTML = '<span class="cv-montage-time">Saving your touches…</span>';
    fetch(cvApiUrl('/api/v2/decisions/' + m.jobId), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decisions: m.decisions, finalize: true })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) { __v2MontageComplete(data.hotspots || [], data.n_hotspots || 0); })
        .catch(function () { __v2MontageComplete([], 0); });
}

function __v2MontageComplete(hotspots, n) {
    saveCvSession({ job_id: __v2Montage ? __v2Montage.jobId : cvJobId, token: cvToken,
                    my_team: cvMyTeamId, state: 'done', v2: true });
    var card = document.querySelector('#cv-montage-screen .cv-montage-card');
    if (!card) return;
    var rows = hotspots.slice(0, 10).map(function (h) {
        return '<div class="cv-montage-hs"><span>' + __v2FmtTime(h.start_sec) + ' – ' +
            __v2FmtTime(h.end_sec) + '</span><span>' + (h.n_contacts || 1) + ' touch' +
            ((h.n_contacts || 1) === 1 ? '' : 'es') + '</span></div>';
    }).join('');
    if (!rows) {
        rows = '<p class="cv-montage-empty">No touches were confirmed as you. Mark more clips ' +
            '“That\'s me”, or add a soccer-specific ball model for better recall.</p>';
    }
    card.innerHTML =
        '<h2 class="cv-overlay-title">Your hotspots are ready</h2>' +
        '<p class="cv-overlay-sub">' + n + ' hotspot' + (n === 1 ? '' : 's') +
        ' built from your confirmed touches.</p>' +
        '<div class="cv-montage-summary">' + rows + '</div>' +
        '<div class="cv-overlay-actions">' +
        '<button type="button" class="cv-btn-secondary" id="cv-montage-back">Back to setup</button>' +
        '<button type="button" class="cv-btn-primary" id="cv-montage-open">Open in workspace</button>' +
        '</div>';
    var backBtn = document.getElementById('cv-montage-back');
    var openBtn = document.getElementById('cv-montage-open');
    if (backBtn) backBtn.onclick = function () {
        document.getElementById('cv-montage-screen').classList.add('hidden');
        var s = document.getElementById('setup-screen'); if (s) s.classList.remove('hidden');
        refreshMatchCatalogue();
    };
    if (openBtn) openBtn.onclick = function () { __v2OpenInWorkspace(hotspots); };
}

function v2HotspotsToSegments(hotspots) {
    // Map v2 hotspots onto the workspace's segment shape (start/end + tick marks).
    return (hotspots || []).map(function (h) {
        return {
            start: h.start_sec,
            end: h.end_sec,
            core_start: h.start_sec,
            core_end: h.end_sec,
            action_triggers: Array.isArray(h.contact_times) ? h.contact_times : []
        };
    });
}

function __v2OpenInWorkspace(hotspots) {
    document.getElementById('cv-montage-screen').classList.add('hidden');
    // Reuse the whole v1 workspace pipeline by handing it v2 hotspots as segments.
    finishCvAnalysis(v2HotspotsToSegments(hotspots), 'v2');
}

// --- 4c. IN-BROWSER DEMO (no server required) ---
function runBrowserDemo() {
    cvSegmentsAreDemo = true;
    const seedVid = document.getElementById('cv-seed-video');
    const duration = (seedVid && isFinite(seedVid.duration) && seedVid.duration > 0)
        ? seedVid.duration : 600;
    const phases = [
        { msg: 'Reading video…', stage: 'init' },
        { msg: 'Loading detector…', stage: 'shot_filter' },
        { msg: 'Scanning for your team colour…', stage: 'deadtime' },
        { msg: 'Tracking the ball…', stage: 'inference' },
        { msg: 'Scoring possession…', stage: 'possession' },
        { msg: 'Merging plays…', stage: 'timestamps' }
    ];
    let p = 0;
    const total = 60;
    const timer = setInterval(function () {
        p += 1;
        const frac = p / total;
        const phaseIdx = Math.min(phases.length - 1, Math.floor(frac * phases.length));
        const phase = phases[phaseIdx];
        setCvProgress({
            progress: frac,
            status: phase.msg + ' (browser demo)',
            stage: phase.stage,
            progress_current: p,
            progress_total: total,
            progress_unit: 'steps'
        });
        if (p >= total) {
            clearInterval(timer);
            const segs = generateDemoSegments(duration);
            finishCvAnalysis(segs, 'browser-demo');
        }
    }, 45);
    // Allow Cancel to abort the demo too.
    cvPollTimer = timer;
}

function generateDemoSegments(duration) {
    // Mirrors server _fake_segments: plausible spread of plays across the match.
    const segs = [];
    let t = 20;
    const horizon = Math.max(60, duration - 15);
    let seed = 7;
    function rnd() { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; }
    while (t < horizon) {
        const start = t + rnd() * 25;
        const len = 8 + rnd() * 14;
        const end = Math.min(duration - 1, start + len);
        if (end <= start) break;
        const triggers = [];
        const nT = 1 + Math.floor(rnd() * 3);
        for (let k = 0; k < nT; k++) triggers.push(+(start + rnd() * (end - start)).toFixed(1));
        triggers.sort(function (a, b) { return a - b; });
        segs.push({ start: +start.toFixed(1), end: +end.toFixed(1), action_triggers: triggers });
        t = end + 15 + rnd() * 35;
    }
    return segs;
}

function pollCvStatus() {
    if (!cvJobId) return;
    cvProcessStart = Date.now();
    let pollFailures = 0;
    const MAX_POLL_FAILURES = 15; // ~12s of consecutive errors before giving up
    cvPollTimer = setInterval(function () {
        fetch(cvApiUrl('/api/process/status/' + cvJobId))
            .then(function (r) { return r.json(); })
            .then(function (j) {
                pollFailures = 0;
                if (j.error && j.state === 'error') {
                    clearInterval(cvPollTimer);
                    // region agent log
                    __dbgJs('B5', 'script.js:pollCvStatus', 'job error', {
                        job_id: cvJobId,
                        error: j.error,
                        stage: j.stage
                    });
                    // endregion
                    cvAnalysisFailed(j.error);
                    return;
                }
                var pollKey = (j.state || '') + '|' + (j.stage || '') + '|' +
                    (j.progress_current != null ? j.progress_current : '') + '/' +
                    (j.progress_total != null ? j.progress_total : '');
                if (pollKey !== cvLastLoggedPollKey) {
                    cvLastLoggedPollKey = pollKey;
                    // region agent log
                    __dbgJs('B3', 'script.js:pollCvStatus', 'poll update', {
                        job_id: cvJobId,
                        state: j.state,
                        stage: j.stage,
                        progress: j.progress,
                        progress_current: j.progress_current,
                        progress_total: j.progress_total,
                        progress_unit: j.progress_unit,
                        elapsed_sec: j.elapsed_sec,
                        status: (j.status || '').slice(0, 120)
                    });
                    // endregion
                }
                setCvProgress(j);
                saveCvSession({
                    job_id: cvJobId,
                    state: j.state,
                    token: j.token || cvToken,
                    my_team: j.my_team || cvMyTeamId
                });
                if (j.state === 'review') {
                    clearInterval(cvPollTimer);
                    enterV2Review(j);
                } else if (j.state === 'done') {
                    clearInterval(cvPollTimer);
                    finishCvAnalysis(j.segments || [], j.note);
                } else if (j.state === 'cancelled') {
                    clearInterval(cvPollTimer);
                    clearCvSession();
                    hideProcessTracker();
                    document.getElementById('cv-processing-screen').classList.add('hidden');
                    const setupScreen = document.getElementById('setup-screen');
                    if (setupScreen) setupScreen.classList.remove('hidden');
                }
            })
            .catch(function (err) {
                pollFailures += 1;
                if (pollFailures < MAX_POLL_FAILURES) {
                    // Transient hiccup — server-side job keeps running; keep polling.
                    setCvProgress({
                        progress: cvTrackerLastPct / 100,
                        status: 'Connection hiccup — retrying (' + pollFailures + ')…',
                        stage: cvTrackerActiveStage
                    });
                    return;
                }
                clearInterval(cvPollTimer);
                cvAnalysisFailed(err.message || String(err));
            });
    }, 800);
}

function cancelCvAnalysis() {
    if (!window.confirm('Stop and discard this analysis run? Saved progress will be deleted.')) {
        return;
    }
    if (cvJobId) {
        fetch(cvApiUrl('/api/process/' + cvJobId), { method: 'DELETE' }).catch(function () {});
    }
    clearCvSession();
    cvToken = null;
    returnToSetupScreen();
}

// Route a status/catalogue payload to the right view (v1 segments vs v2 review/hotspots).
function finishFromStatus(j) {
    if (j && j.pipeline_version === 'v2') {
        if (j.state === 'review') { enterV2Review(j); return; }
        pfShowPipelineWarnings(j.warnings || []);   // review was skipped/empty — still surface these
        finishCvAnalysis(v2HotspotsToSegments(j.hotspots || []), 'v2');
        return;
    }
    finishCvAnalysis((j && j.segments) || [], j && j.note);
}

function finishCvAnalysis(segments, note, opts) {
    opts = opts || {};
    clearReviewSession();
    clipSegmentsLibrary = Array.isArray(segments) ? segments : [];
    if (note === 'browser-demo' || (note && note.indexOf('demo') !== -1)) cvSegmentsAreDemo = true;
    // region agent log
    __dbgJs('H1', 'script.js:finishCvAnalysis', 'analysis done', {
        demo: cvSegmentsAreDemo,
        note: note || '',
        n_segments: clipSegmentsLibrary.length,
        last_end: clipSegmentsLibrary.length ? clipSegmentsLibrary[clipSegmentsLibrary.length - 1].end : null
    });
    // endregion
    updateProcessTracker({
        stage: 'done',
        progress: 100,
        status: 'Analysis complete — loading match view…'
    });
    document.getElementById('cv-processing-screen').classList.add('hidden');
    const setupScreen = document.getElementById('setup-screen');
    if (setupScreen) setupScreen.classList.add('hidden');
    if (note) console.log('[PolyFut-CV] ' + note);
    if (!clipSegmentsLibrary.length && !cvSegmentsAreDemo) {
        console.warn('[PolyFut-CV] Analysis finished with no touch hotspots.');
    }
    setTimeout(hideProcessTracker, 2500);
    saveCvSession({ state: 'done', job_id: cvJobId, token: cvToken });
    var meta = getSetupMetadataFields();
    if (!opts.skipCatalogue) {
        var catEntry = {
            job_id: cvJobId,
            token: cvToken,
            my_team: cvMyTeamId,
            opponent: meta.opponent,
            match_date: meta.match_date,
            score_us: meta.score_us,
            score_them: meta.score_them,
            position: meta.position,
            n_hotspots: clipSegmentsLibrary.length,
            analysed_at: Date.now() / 1000,
            // Only a server-uploaded token can be replayed later — a manual
            // session's video is a local blob URL that dies with the tab.
            video_available: !!cvToken
        };
        pushMatchCatalogueEntry(catEntry);
    }
    sendMatchMetadataToServer(cvJobId, meta);
    var videoSrc = cvVideoURL || cvVideoUrlForToken(cvToken);
    enterMainAppWithVideo(videoSrc);
    attachSessionRestoreOnVideoReady(cvJobId);
}

function cvAnalysisFailed(msg) {
    updateProcessTracker({
        stage: cvTrackerActiveStage,
        progress: 18,
        status: 'Analysis failed — continuing without touch hotspots',
        state: 'error'
    });
    setCvProgress({ progress: 0, status: 'Analyser unreachable. Continuing without touch hotspots.', stage: 'error' });
    console.warn('[PolyFut-CV] ' + msg);
    setTimeout(function () {
        clipSegmentsLibrary = [];
        document.getElementById('cv-processing-screen').classList.add('hidden');
        const setupScreen = document.getElementById('setup-screen');
        if (setupScreen) setupScreen.classList.add('hidden');
        setTimeout(hideProcessTracker, 2000);
        var videoSrc = cvVideoURL || cvVideoUrlForToken(cvToken);
        enterMainAppWithVideo(videoSrc);
    }, 1800);
}

function mapServerStageToTracker(stage) {
    var known = ['upload', 'kits', 'init', 'shot_filter', 'deadtime', 'inference', 'possession', 'timestamps', 'done', 'error', 'cancelled'];
    if (known.indexOf(stage) >= 0) return stage === 'error' || stage === 'cancelled' ? cvTrackerActiveStage : stage;
    if (stage === 'running') return 'inference';
    return stage || 'inference';
}

function serverProgressToOverall(serverFrac, stage) {
    if (stage === 'done') return 100;
    return Math.round(18 + (serverFrac || 0) * 82);
}

function setCvProgress(jobOrPct, statusText) {
    const bar = document.getElementById('cv-progress-bar');
    const st = document.getElementById('cv-processing-status');
    const detail = document.getElementById('cv-processing-detail');
    let pct = 0;
    let status = '';
    let stage = '';
    let elapsed = '';
    let trackerOpts = null;
    if (typeof jobOrPct === 'object' && jobOrPct) {
        pct = (jobOrPct.progress || 0) * 100;
        status = jobOrPct.status || 'Analyzing...';
        stage = jobOrPct.stage || '';
        elapsed = jobOrPct.elapsed_sec != null ? jobOrPct.elapsed_sec + 's' : '';
        var trkStage = mapServerStageToTracker(stage || (jobOrPct.state === 'done' ? 'done' : 'inference'));
        trackerOpts = {
            progress: serverProgressToOverall(jobOrPct.progress || 0, trkStage),
            status: status,
            stage: trkStage,
            elapsed_sec: jobOrPct.elapsed_sec,
            segments_partial: jobOrPct.segments_partial,
            progress_current: jobOrPct.progress_current,
            progress_total: jobOrPct.progress_total,
            progress_unit: jobOrPct.progress_unit,
            state: jobOrPct.state === 'error' ? 'error' : undefined
        };
    } else {
        pct = jobOrPct || 0;
        status = statusText || '';
        trackerOpts = {
            progress: Math.round(18 + (pct / 100) * 82),
            status: status,
            stage: pct >= 99 ? 'done' : mapServerStageToTracker(cvTrackerActiveStage)
        };
    }
    if (bar) bar.style.width = Math.max(0, Math.min(100, pct)) + '%';
    if (st && status) st.innerText = status;
    if (detail) {
        const parts = [];
        if (stage) parts.push('Stage: ' + stage);
        if (elapsed) parts.push('Elapsed: ' + elapsed);
        parts.push(Math.round(pct) + '%');
        detail.innerText = parts.join(' · ');
    }
    if (trackerOpts) updateProcessTracker(trackerOpts);
}

// --- 5. VIDEO TIMER ---
function updateVideoTimer() {
    if (isSeeking) return;
    const videoPlayer = document.getElementById('main-player');
    const slider = document.getElementById('seek-slider');
    const timeDisplay = document.getElementById('time-display');
    const placeholder = document.getElementById('vid-placeholder');

    if (videoPlayer && videoPlayer.currentTime > 0 && placeholder && placeholder.style.display !== 'none') {
        placeholder.style.display = 'none';
        if (slider) slider.max = videoPlayer.duration;
    }

    slider.value = videoPlayer.currentTime;
    const m = Math.floor(videoPlayer.currentTime / 60).toString().padStart(2, '0');
    const s = Math.floor(videoPlayer.currentTime % 60).toString().padStart(2, '0');
    if (timeDisplay) timeDisplay.innerText = `${m}:${s}`;
}

// --- 6. LOG STATS ---
function logStat(actionName) {
    const videoPlayer = document.getElementById('main-player');
    const currentTime = videoPlayer.currentTime;

    const isBenched = benchBlocks.some(b => {
        const start = b.startPct * videoPlayer.duration;
        const end = b.endPct * videoPlayer.duration;
        return currentTime >= start && currentTime <= end;
    });

    if (isBenched) {
        alert("Cannot log stats while on the bench!");
        return;
    }

    // Note: logging is allowed outside hotspots too (FREE PLAY) — the CV is
    // recall-biased and can miss touches; users must be able to log them.

    const m = Math.floor(currentTime / 60).toString().padStart(2, '0');
    const s = Math.floor(currentTime % 60).toString().padStart(2, '0');
    matchStats.push({ action: actionName, timeStr: `${m}:${s}`, seconds: currentTime });

    if (typeof calculatePerformance === "function") {
        const liveResults = calculatePerformance(
            matchStats, currentScore, videoPlayer.duration || 90,
            getAllExcludedRanges(videoPlayer.duration || 90), selectedPosition || 'FW'
        );
        const elNet = document.getElementById('dash-net');
        if (elNet) {
            elNet.innerText = liveResults.netScore;
            elNet.style.color = parseFloat(liveResults.netScore) >= 0 ? '#4caf50' : '#ff2e4d';
        }
        const elOff = document.getElementById('dash-off-markov');
        if (elOff) elOff.innerText = liveResults.offMarkov;
        const elDef = document.getElementById('dash-def-markov');
        if (elDef) elDef.innerText = liveResults.defMarkov;
        const elRisk = document.getElementById('dash-risk');
        if (elRisk) {
            const totalRisk = (parseFloat(liveResults.offRidge) + parseFloat(liveResults.defRidge)).toFixed(3);
            elRisk.innerText = totalRisk;
        }
    }

    const app = document.getElementById('app-layout');
    app.style.boxShadow = "inset 0 0 20px #30ff8f";
    setTimeout(() => { app.style.boxShadow = "none"; }, 150);
    if (navigator.vibrate) navigator.vibrate(50);
    scheduleSaveMatchSession();
}

// --- 7. BENCH LOGIC ---
function addBenchBlock() {
    const block = createBenchBlock(Date.now(), 0.4, 0.5);
    if (block) {
        benchBlocks.push(block);
        scheduleSaveMatchSession();
    }
}

function removeBenchBlock(id) {
    const index = benchBlocks.findIndex(b => b.id === id);
    if (index > -1) { benchBlocks[index].element.remove(); benchBlocks.splice(index, 1); }
    scheduleSaveMatchSession();
}

function setupBlockListeners(blockObj, leftBtn, rightBtn) {
    const track = document.getElementById('bench-track');
    const onEdgeDrag = (e, isLeft) => {
        const rect = track.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let pct = Math.max(0, Math.min(1, x / rect.width));
        if (isLeft) blockObj.startPct = Math.min(pct, blockObj.endPct - 0.01);
        else blockObj.endPct = Math.max(pct, blockObj.startPct + 0.01);
        renderBlock(blockObj);
    };
    const startEdgeDrag = (e, isLeft) => {
        e.preventDefault();
        const move = (ev) => onEdgeDrag(ev, isLeft);
        const stop = () => {
            window.removeEventListener('mousemove', move);
            window.removeEventListener('mouseup', stop);
            scheduleSaveMatchSession();
        };
        window.addEventListener('mousemove', move); window.addEventListener('mouseup', stop);
    };
    leftBtn.addEventListener('mousedown', (e) => startEdgeDrag(e, true));
    rightBtn.addEventListener('mousedown', (e) => startEdgeDrag(e, false));

    const fillBtn = blockObj.element.querySelector('.bench-fill');
    fillBtn.addEventListener('mousedown', (e) => {
        e.preventDefault();
        const rect = track.getBoundingClientRect();
        let startX = e.clientX;
        let initialStartPct = blockObj.startPct;
        let initialEndPct = blockObj.endPct;
        let blockWidthPct = initialEndPct - initialStartPct;
        const moveBlock = (ev) => {
            let dx = ev.clientX - startX;
            let dPct = dx / rect.width;
            let newStart = initialStartPct + dPct;
            let newEnd = initialEndPct + dPct;
            if (newStart < 0) { newStart = 0; newEnd = blockWidthPct; }
            else if (newEnd > 1) { newEnd = 1; newStart = 1 - blockWidthPct; }
            blockObj.startPct = newStart; blockObj.endPct = newEnd;
            renderBlock(blockObj);
        };
        const stopBlock = () => {
            window.removeEventListener('mousemove', moveBlock);
            window.removeEventListener('mouseup', stopBlock);
            scheduleSaveMatchSession();
        };
        window.addEventListener('mousemove', moveBlock); window.addEventListener('mouseup', stopBlock);
    });
}

function renderBlock(blockObj) {
    const leftH = blockObj.element.querySelector('.left');
    const rightH = blockObj.element.querySelector('.right');
    const fill = blockObj.element.querySelector('.bench-fill');
    const remove = blockObj.element.querySelector('.bench-remove');
    leftH.style.left = (blockObj.startPct * 100) + '%';
    rightH.style.left = (blockObj.endPct * 100) + '%';
    fill.style.left = (blockObj.startPct * 100) + '%';
    fill.style.width = ((blockObj.endPct - blockObj.startPct) * 100) + '%';
    remove.style.left = ((blockObj.startPct + blockObj.endPct) / 2 * 100) + '%';
}

// --- 8. VIDEO CONTROLS ---
function updateSpeedConfig() {
    const slowInput = document.getElementById('slow-speed-set');
    const fastInput = document.getElementById('fast-speed-set');
    if (slowInput) slowSpeed = parseFloat(slowInput.value);
    if (fastInput) fastSpeed = parseFloat(fastInput.value);
}

function toggleCustomSpeed() {
    const videoPlayer = document.getElementById('main-player');
    if (videoPlayer.playbackRate <= slowSpeed) setSpeed(fastSpeed);
    else setSpeed(slowSpeed);
}

function setSpeed(rate) {
    const videoPlayer = document.getElementById('main-player');
    const display = document.getElementById('current-speed-display');
    videoPlayer.playbackRate = rate;
    if (rate > 2) videoPlayer.muted = true; else videoPlayer.muted = false;
    if (display) display.innerText = rate + 'x';
}

function handleKeyShortcuts(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const videoPlayer = document.getElementById('main-player');
    if (!videoPlayer) return;
    switch (e.code) {
        case 'Space': e.preventDefault(); togglePlay(); break;
        case 'ArrowLeft': videoPlayer.currentTime = clampClipSeek(Math.max(0, videoPlayer.currentTime - 5)); break;
        case 'ArrowRight': videoPlayer.currentTime = clampClipSeek(Math.min(videoPlayer.duration, videoPlayer.currentTime + 5)); break;
        case 'KeyS': toggleCustomSpeed(); break;
    }
}

function togglePlay() {
    const video = document.getElementById('main-player');
    const btn = document.getElementById('play-pause-btn');
    if (video.paused) { video.play(); btn.innerText = "⏸"; }
    else { video.pause(); btn.innerText = "▶"; }
}

// --- 9. ZOOM & PAN ---
function applyTransform() {
    const video = document.getElementById('main-player');
    video.style.transform = `scale(${zoomLevel}) translate(${panX}px, ${panY}px)`;
    document.getElementById('video-wrapper').style.cursor = zoomLevel > 1 ? 'grab' : 'default';
}
function handleWheel(e) {
    e.preventDefault();
    if (e.deltaY < 0) zoomLevel += 0.1; else zoomLevel -= 0.1;
    zoomLevel = Math.min(Math.max(1, zoomLevel), 5);
    if (zoomLevel === 1) { panX = 0; panY = 0; }
    applyTransform();
}
function startPan(e) {
    if (zoomLevel <= 1) return;
    isDragging = true;
    startX = e.clientX - panX * zoomLevel;
    startY = e.clientY - panY * zoomLevel;
    document.getElementById('video-wrapper').style.cursor = 'grabbing';
}
function pan(e) {
    if (!isDragging) return;
    e.preventDefault();
    panX = (e.clientX - startX) / zoomLevel;
    panY = (e.clientY - startY) / zoomLevel;
    applyTransform();
}
function endPan() {
    isDragging = false;
    if (zoomLevel > 1) document.getElementById('video-wrapper').style.cursor = 'grab';
}
function resetZoom() { zoomLevel = 1; panX = 0; panY = 0; applyTransform(); }

// --- 10. MENUS ---
function openSubMenu(type) {
    const mainMenu = document.getElementById('menu-main');
    mainMenu.style.display = 'none';
    mainMenu.classList.add('hidden');
    const targetMenu = document.getElementById('menu-' + type);
    targetMenu.classList.remove('hidden');
    targetMenu.style.display = 'grid';
}

function goBack() {
    document.querySelectorAll('.sub-menu').forEach(el => { el.style.display = 'none'; el.classList.add('hidden'); });
    const mainMenu = document.getElementById('menu-main');
    mainMenu.classList.remove('hidden');
    mainMenu.style.display = 'grid';
}

// --- 11. FINISH MATCH ---
let currentHybridResults = null;

// Colors a stat span by sign: green if it worked out for the player this
// match, red if it didn't. Used for Impact/Risk, which are net totals that can
// land on either side of zero (unlike Goals/Assists, which only ever add up).
function setSignedStat(elId, value) {
    const el = document.getElementById(elId);
    if (!el) return;
    el.innerText = value;
    el.style.color = parseFloat(value) >= 0 ? 'var(--fifa-green)' : 'var(--fifa-red)';
}

function finishMatch() {
    const videoPlayer = document.getElementById('main-player');
    videoPlayer.pause();
    const duration = videoPlayer.duration || 0;
    let excludedRanges = getAllExcludedRanges(duration);

    document.getElementById('app-layout').classList.add('hidden');
    document.getElementById('app-layout').style.display = 'none';
    const resScreen = document.getElementById('results-screen');
    resScreen.classList.remove('hidden');
    resScreen.style.display = 'flex';

    if (typeof calculatePerformance === "function") {
        currentHybridResults = calculatePerformance(matchStats, currentScore, duration, excludedRanges, selectedPosition);
        document.getElementById('result-header').innerText = `PERFORMANCE REPORT (${selectedPosition})`;
        const totalGoals = matchStats.filter(s => s.action === 'Goal').length;
        const totalAssists = matchStats.filter(s => s.action === 'Assist').length;
        document.getElementById('res-goals').innerText = totalGoals;
        document.getElementById('res-assists').innerText = totalAssists;
        document.getElementById('res-overall').innerText = currentHybridResults.netScore;
        const oaVal = parseFloat(currentHybridResults.netScore);
        document.getElementById('res-overall').style.color = oaVal >= 0 ? '#4caf50' : '#f44336';

        // Impact and Risk are both net figures that can land either side of zero
        // (e.g. a dispossession drags Impact negative; a risk that didn't pay
        // off drags Risk negative) — color by sign so "did this help or hurt"
        // reads at a glance instead of hiding inside a plain number.
        setSignedStat('res-off-markov', currentHybridResults.offMarkov);
        setSignedStat('res-off-ridge', currentHybridResults.offRidge);
        setSignedStat('res-def-markov', currentHybridResults.defMarkov);
        setSignedStat('res-def-ridge', currentHybridResults.defRidge);
        renderWPAChart(currentHybridResults.chartData, duration, excludedRanges);
        saveMatchSession(cvJobId);
    } else {
        alert("Error: calculations.js is not loaded correctly.");
    }
}

// --- 12. CHARTING ---
function renderWPAChart(data, maxDuration, excludedRanges) {
    const ctx = document.getElementById('wpaChart').getContext('2d');
    const annotations = {};
    excludedRanges.forEach((range, index) => {
        annotations['box' + index] = {
            type: 'box', xMin: range.start, xMax: range.end,
            backgroundColor: 'rgba(255, 23, 68, 0.2)', borderColor: 'transparent',
            label: { display: true, content: 'BENCH', color: 'rgba(255,255,255,0.5)', font: { size: 10 } }
        };
    });
    const formatTime = (seconds) => {
        const m = Math.floor(seconds / 60).toString().padStart(2, '0');
        const s = Math.floor(seconds % 60).toString().padStart(2, '0');
        return `${m}:${s}`;
    };
    if (window.myChart) window.myChart.destroy();
    window.myChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Threat Points (TP)', data: data,
                borderColor: '#30ff8f', backgroundColor: 'rgba(48, 255, 143, 0.1)',
                borderWidth: 2, fill: true, tension: 0.1, pointRadius: 0
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { title: (c) => formatTime(c[0].parsed.x) } },
                annotation: { annotations: annotations }
            },
            scales: {
                y: { grid: { color: '#222' }, ticks: { color: '#666' } },
                x: { type: 'linear', position: 'bottom', max: maxDuration, grid: { display: false }, ticks: { color: '#666', callback: (v) => formatTime(v) } }
            }
        }
    });
}

// --- 13. AI SCOUT REPORT (GROQ) ---
const PF_GROQ_KEY = 'futfidget_groq_key';
const PF_ONBOARDING_DONE_KEY = 'polyfut_onboarding_done';

// --- AI transport: hosted proxy first, user's own key as the fallback -------
// Reports used to require every user to make a Groq account and paste a key.
// The key now lives in a Modal secret behind a proxy we host (see ai_backend/),
// so the normal path needs nothing from the user. A locally-saved key still
// works and is used automatically if the proxy is unconfigured or unreachable —
// so a Modal outage or an exhausted daily quota degrades to the old behaviour
// instead of removing the feature.
let pfAiProxy = null;          // {proxy_url, app_token} once /api/ai_config answers
let pfAiProxyChecked = false;

function pfLoadAiConfig() {
    return fetch(cvApiUrl('/api/ai_config'))
        .then(function (r) { return r.json(); })
        .then(function (cfg) {
            pfAiProxy = (cfg && cfg.enabled && cfg.proxy_url) ? cfg : null;
            return pfAiProxy;
        })
        .catch(function () { pfAiProxy = null; return null; })
        .then(function (out) { pfAiProxyChecked = true; return out; });
}

function pfAiReady() {
    // Either transport counts: the hosted proxy, or a key the user saved.
    try { return !!pfAiProxy || !!localStorage.getItem(PF_GROQ_KEY); } catch (e) { return !!pfAiProxy; }
}

function pfSavedGroqKey() {
    try {
        return localStorage.getItem(PF_GROQ_KEY) ||
            (document.getElementById('api-key-input') || {}).value || '';
    } catch (e) { return ''; }
}

/**
 * Send a chat history and get the assistant's reply text.
 *
 * One helper for both the initial report and the follow-up Q&A, because both
 * are the same call with a different message list — the proxy takes the whole
 * history so multi-turn conversation survives the move off direct Groq calls.
 * Throws an Error whose message is safe to show the user.
 */
async function pfAiChat(messages, opts) {
    opts = opts || {};
    if (!pfAiProxyChecked) await pfLoadAiConfig();

    if (pfAiProxy) {
        let data, response;
        try {
            response = await fetch(pfAiProxy.proxy_url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    app_token: pfAiProxy.app_token,
                    messages: messages,
                    temperature: opts.temperature != null ? opts.temperature : 0.6,
                    response_format: opts.response_format
                })
            });
            data = await response.json();
        } catch (e) {
            // Network-level failure only — fall through to the user's own key if
            // they have one, otherwise report it.
            if (!pfSavedGroqKey()) {
                throw new Error('Could not reach the AI service. Check your internet connection.');
            }
            return pfAiChatDirect(messages, opts);
        }
        if (response.ok && data && data.report) return data.report;
        // A 4xx/5xx from the proxy is a real, explained failure (bad token, rate
        // limit, upstream error) — surface it rather than silently retrying,
        // unless the user has their own key to fall back on.
        var msg = (data && data.error) || ('AI service error (HTTP ' + response.status + ')');
        // Two different things arrive as 429. scope "app" is the shared
        // endpoint's own per-IP allowance, and using a personal key there would
        // route around our own abuse control — so that one is reported, not
        // retried. scope "upstream" is the shared Groq quota being exhausted,
        // which is precisely the case a user's own key should cover; without
        // this, someone who had gone to the trouble of adding a key still got
        // told the service was busy.
        // Only an explicit "upstream" unlocks the personal key. A proxy too old
        // to send scope says nothing, and the safe reading of silence is the
        // app's own limit — better to report a busy service than to quietly
        // hand everyone a way around it.
        var appLimited = response.status === 429
            && !(data && data.scope === 'upstream');
        if (pfSavedGroqKey() && !appLimited) return pfAiChatDirect(messages, opts);
        throw new Error(msg);
    }

    return pfAiChatDirect(messages, opts);
}

// Legacy path: call Groq directly with a key the user pasted in themselves.
async function pfAiChatDirect(messages, opts) {
    const apiKey = pfSavedGroqKey();
    if (!apiKey) {
        throw new Error('AI reports are unavailable right now. Please try again later.');
    }
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + apiKey },
        body: JSON.stringify({
            model: 'llama-3.3-70b-versatile',
            messages: messages,
            temperature: (opts && opts.temperature != null) ? opts.temperature : 0.6,
            response_format: opts && opts.response_format
        })
    });
    const data = await response.json();
    if (data.error) throw new Error(data.error.message);
    if (!data.choices || !data.choices.length) throw new Error('The AI returned an empty response.');
    return data.choices[0].message.content;
}

document.addEventListener('DOMContentLoaded', () => {
    const savedKey = localStorage.getItem(PF_GROQ_KEY);
    if (savedKey) toggleKeyUI(true);
    // The key UI and the "paste a key" onboarding are only worth showing when
    // there's no proxy — otherwise we'd be asking for something we don't need.
    pfLoadAiConfig().then(function () {
        if (pfAiProxy) pfHideKeyUiForProxy();
        pfMaybeShowOnboarding();
    });
});

// Hide the bring-your-own-key surfaces when the hosted proxy is available. The
// code paths stay live — a saved key is still honoured as a fallback — this
// only stops us asking users for a key they don't need.
function pfHideKeyUiForProxy() {
    ['api-key-container', 'api-key-saved'].forEach(function (id) {
        const el = document.getElementById(id);
        if (el) { el.classList.add('hidden'); el.style.display = 'none'; }
    });
    pfMarkOnboardingDone();   // never prompt for a key on first launch
}

function toggleKeyUI(isSaved) {
    const inputContainer = document.getElementById('api-key-container');
    const savedBadge = document.getElementById('api-key-saved');
    if (isSaved) {
        if (inputContainer) { inputContainer.classList.add('hidden'); inputContainer.style.display = 'none'; }
        if (savedBadge) { savedBadge.classList.remove('hidden'); savedBadge.style.display = 'flex'; }
    } else {
        if (inputContainer) { inputContainer.classList.remove('hidden'); inputContainer.style.display = 'flex'; }
        if (savedBadge) { savedBadge.classList.add('hidden'); savedBadge.style.display = 'none'; }
    }
}

function saveGroqApiKey(rawKey, { quiet } = {}) {
    const key = String(rawKey || '').trim();
    if (!key.startsWith('gsk_')) {
        return { ok: false, error: "Invalid key. Groq API keys usually start with 'gsk_'." };
    }
    try {
        localStorage.setItem(PF_GROQ_KEY, key);
    } catch (e) {
        return { ok: false, error: 'Could not save the key on this device.' };
    }
    toggleKeyUI(true);
    const resultsInput = document.getElementById('api-key-input');
    if (resultsInput) resultsInput.value = '';
    if (!quiet) alert('Key saved securely to your browser!');
    return { ok: true };
}

function saveApiKey() {
    const input = document.getElementById('api-key-input');
    const result = saveGroqApiKey(input ? input.value : '');
    if (!result.ok) alert(result.error);
}

function clearApiKey() {
    localStorage.removeItem(PF_GROQ_KEY);
    const input = document.getElementById('api-key-input');
    if (input) input.value = '';
    toggleKeyUI(false);
}

function pfOnboardingDone() {
    try { return localStorage.getItem(PF_ONBOARDING_DONE_KEY) === '1'; } catch (e) { return false; }
}

function pfMarkOnboardingDone() {
    try { localStorage.setItem(PF_ONBOARDING_DONE_KEY, '1'); } catch (e) {}
}

function pfCloseOnboarding() {
    const screen = document.getElementById('pf-onboarding-screen');
    if (screen) screen.classList.add('hidden');
    pfMarkOnboardingDone();
}

function pfMaybeShowOnboarding() {
    // Already finished/skipped, or a key is already saved → never block setup.
    try {
        if (pfOnboardingDone()) return;
        if (localStorage.getItem(PF_GROQ_KEY)) {
            pfMarkOnboardingDone();
            return;
        }
    } catch (e) { return; }

    const screen = document.getElementById('pf-onboarding-screen');
    if (!screen) return;
    screen.classList.remove('hidden');

    const err = document.getElementById('pf-onboarding-error');
    const input = document.getElementById('pf-onboarding-key');
    const saveBtn = document.getElementById('pf-onboarding-save');
    const skipBtn = document.getElementById('pf-onboarding-skip');

    function showErr(msg) {
        if (!err) return;
        if (msg) {
            err.textContent = msg;
            err.classList.remove('hidden');
        } else {
            err.textContent = '';
            err.classList.add('hidden');
        }
    }

    if (saveBtn && !saveBtn._pfBound) {
        saveBtn._pfBound = true;
        saveBtn.addEventListener('click', function () {
            const result = saveGroqApiKey(input ? input.value : '', { quiet: true });
            if (!result.ok) { showErr(result.error); return; }
            showErr('');
            pfCloseOnboarding();
        });
    }
    if (skipBtn && !skipBtn._pfBound) {
        skipBtn._pfBound = true;
        skipBtn.addEventListener('click', function () {
            showErr('');
            pfCloseOnboarding();
        });
    }
    if (input && !input._pfBound) {
        input._pfBound = true;
        input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') saveBtn && saveBtn.click();
        });
    }
}

// The 8 body sections the report is split into (everything but the closing
// verdict, which renders as its own banner instead of a card).
const SCOUT_SECTIONS = [
    { key: 'tactical_role', label: 'Tactical role' },
    { key: 'key_strengths', label: 'Key strengths' },
    { key: 'areas_to_improve', label: 'Areas to improve' },
    { key: 'risks', label: 'Risks in play style' },
    { key: 'drills', label: 'Drills' },
    { key: 'mentality', label: 'Mentality shift' },
    { key: 'temporal', label: 'Temporal read' },
    { key: 'other_insights', label: 'Other insights' },
];

function escapeHtml(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Renders the parsed {tactical_role, key_strengths, ..., summary} object as
// the card-grid document (Layout B). Missing/blank keys are skipped rather
// than shown empty, since a real model response occasionally drops one.
function renderScoutReport(data, position, netScore) {
    const cards = SCOUT_SECTIONS
        .filter(s => data[s.key] && String(data[s.key]).trim())
        .map(s => `<div class="scout-card"><h4>${escapeHtml(s.label)}</h4><p>${escapeHtml(data[s.key])}</p></div>`)
        .join('');
    const verdict = data.summary && String(data.summary).trim()
        ? `<div class="scout-verdict"><p class="scout-verdict-lbl">Verdict</p><p>${escapeHtml(data.summary)}</p></div>`
        : '';
    return `
        <div class="scout-top">
            <div><span class="scout-score">${escapeHtml(String(netScore))}</span><div class="scout-score-lbl">Net TP</div></div>
            <p>${escapeHtml(position)} — full scouting breakdown below.</p>
        </div>
        <div class="scout-grid">${cards}</div>
        ${verdict}
        <div class="scout-thread" id="scout-thread"></div>
    `;
}

// Fallback for when the model doesn't return valid JSON (rare, but response_format
// is a request, not a guarantee) — show the raw text rather than losing the report.
function renderScoutReportFallback(rawText) {
    return `
        <div class="scout-card"><p style="white-space: pre-wrap;">${escapeHtml(rawText)}</p></div>
        <div class="scout-thread" id="scout-thread"></div>
    `;
}

async function generateScoutReport() {
    const outputBox = document.getElementById('ai-output');
    const btn = document.getElementById('gen-report-btn');
    if (!pfAiProxyChecked) await pfLoadAiConfig();
    if (!pfAiReady()) { alert("AI reports aren't available right now. Please try again later."); return; }
    if (matchStats.length === 0 || !currentHybridResults) { outputBox.innerHTML = "<p class='scout-placeholder'>No stats collected yet. Play a match first!</p>"; return; }

    btn.disabled = true;
    btn.innerText = "SCOUTING...";
    outputBox.innerHTML = "<p class='scout-placeholder' style='color:#f2c94c'>Analyzing gameplay patterns & calculating hybrid matrices...</p>";

    const statSummary = matchStats.reduce((acc, curr) => { acc[curr.action] = (acc[curr.action] || 0) + 1; return acc; }, {});
    const timelineString = matchStats.map(s => `[${s.timeStr}] ${s.action}`).join(', ');
    const position = selectedPosition || "Player";
    const totalMarkov = (parseFloat(currentHybridResults.offMarkov) + parseFloat(currentHybridResults.defMarkov)).toFixed(2);
    const totalRidge = (parseFloat(currentHybridResults.offRidge) + parseFloat(currentHybridResults.defRidge)).toFixed(2);
    const netScore = currentHybridResults.netScore;
    const markovValuations = JSON.stringify(currentHybridResults.coeffMarkov);
    const ridgeValuations = JSON.stringify(currentHybridResults.coeffRidge);

    const systemPrompt = `You are a professional Premier League scout.

    Analyze the following player stats for a single match.
    The player position is ${position}. Their final score was ${netScore} Threat Points (TP). Their direct contributions were ${totalMarkov} TP (Markov), however the long term impacts of their actions and their risks could actually be ${totalRidge} TP (Ridge).

    Scores are in Threat Points: 1 TP = 1% of a goal (so a goal is worth 100 TP). The score is a hybrid valuation from Markov matrices (immediate chain impact) and Ridge Regression (long-term win/loss signal). Progression may help on the Markov level but carry Ridge risks if excess volume leads to turnovers. A Goal auto-suppresses a preceding Shot Taken so goals are not double-counted.

    Here is the action count: ${JSON.stringify(statSummary)}.

    To help you analyze risk vs reward, here are relative action importance labels for a ${position} (not exact numeric weights):
    - Markov (Immediate threat): ${markovValuations}
    - Ridge (Long term Win/Loss impact): ${ridgeValuations}

    Here is the chronological timeline of their actions: ${timelineString}.

    Respond with ONLY a single valid JSON object (no markdown fences, no commentary) with exactly these string keys:
    - tactical_role: their role based on actions
    - key_strengths: strengths shown by high-count actions
    - areas_to_improve: missing actions expected for this position
    - risks: whether their action volume creates unnecessary risk
    - drills: drills to address the weaknesses above
    - mentality: a mentality change that would help them improve
    - temporal: how their performance changed over the match (first half vs second half, time on the bench, why they may have been subbed)
    - other_insights: anything else notable
    - summary: a single-sentence overall rating

    Each value should be 1-3 sentences, professional and critical in tone, under 500 words total across all fields combined.
    (IMPORTANT DIRECTIVE: this strict JSON format is for this first report only — any follow-up question after this one should be answered directly, in plain prose, not as JSON.)
    Additionally, if the player asks anything that isn't related to football and their improvement, please respond that you can't answer because you are an AI strictly used to coach football.`;

    try {
        aiChatHistory = [
            { role: "system", content: "You are a highly analytical football data scientist and scout." },
            { role: "user", content: systemPrompt }
        ];
        const report = await pfAiChat(aiChatHistory, { temperature: 0.6, response_format: { type: 'json_object' } });
        aiChatHistory.push({ role: "assistant", content: report });

        let parsed = null;
        try { parsed = JSON.parse(report); } catch (e) { /* handled below */ }
        if (parsed && typeof parsed === 'object') {
            outputBox.innerHTML = renderScoutReport(parsed, position, netScore);
        } else {
            console.warn('Scout report was not valid JSON, showing raw text instead.');
            outputBox.innerHTML = renderScoutReportFallback(report);
        }

        document.getElementById('followup-container').classList.remove('hidden');
        btn.innerText = "REPORT GENERATED";
        btn.style.background = "#2e7d32";
        setTimeout(() => { btn.disabled = false; btn.innerText = "GENERATE AGAIN"; btn.style.background = "linear-gradient(135deg, #7000ff 0%, #3d0096 100%)"; }, 3000);
    } catch (error) {
        console.error(error);
        outputBox.innerHTML = `<p class="scout-placeholder" style="color:#ff2e4d">Error: ${escapeHtml(error.message)}</p>`;
        btn.innerText = "TRY AGAIN";
        btn.disabled = false;
        // Only wipe a saved key when WE were the ones using it — on the hosted
        // proxy a 401 means our server-side credentials are wrong, and clearing
        // the user's key would be both useless and confusing.
        if (!pfAiProxy && (error.message.includes("401") || error.message.includes("key"))) {
            clearApiKey();
            alert("Your API Key seems invalid. Please check it.");
        }
    }
}

// --- 15. FOLLOW-UP Q&A ---
async function askFollowUp() {
    const inputField = document.getElementById('followup-input');
    const question = inputField.value.trim();
    const outputBox = document.getElementById('ai-output');
    const askBtn = document.getElementById('followup-btn');
    if (!question || !pfAiReady()) return;

    // The thread lives inside #ai-output, appended after the cards/verdict —
    // generateScoutReport() always leaves one behind, even on the JSON-parse
    // fallback path, but guard for the case nothing has been generated yet.
    let thread = document.getElementById('scout-thread');
    if (!thread) {
        thread = document.createElement('div');
        thread.className = 'scout-thread';
        thread.id = 'scout-thread';
        outputBox.appendChild(thread);
    }

    aiChatHistory.push({ role: "user", content: question });
    const youMsg = document.createElement('div');
    youMsg.className = 'scout-msg you';
    youMsg.innerHTML = `<span class="scout-msg-lbl">You</span>`;
    youMsg.appendChild(document.createTextNode(question));
    thread.appendChild(youMsg);

    const aiMsg = document.createElement('div');
    aiMsg.className = 'scout-msg ai';
    aiMsg.innerHTML = `<span class="scout-msg-lbl">Coach AI</span><em id="ai-thinking">Thinking...</em>`;
    thread.appendChild(aiMsg);

    outputBox.scrollTop = outputBox.scrollHeight;
    inputField.value = "";
    askBtn.disabled = true;
    askBtn.style.opacity = "0.5";

    try {
        const answer = await pfAiChat(aiChatHistory, { temperature: 0.6 });
        aiChatHistory.push({ role: "assistant", content: answer });
        const thinkingEl = document.getElementById('ai-thinking');
        if (thinkingEl) { const span = document.createElement('span'); span.textContent = answer; thinkingEl.replaceWith(span); }
    } catch (error) {
        console.error(error);
        const thinkingEl = document.getElementById('ai-thinking');
        if (thinkingEl) { const errSpan = document.createElement('span'); errSpan.style.color = '#ff2e4d'; errSpan.textContent = 'Error: ' + error.message; thinkingEl.replaceWith(errSpan); }
    } finally {
        askBtn.disabled = false;
        askBtn.style.opacity = "1";
        outputBox.scrollTop = outputBox.scrollHeight;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const fInput = document.getElementById('followup-input');
    if (fInput) fInput.addEventListener('keypress', function (e) { if (e.key === 'Enter') askFollowUp(); });
});

// --- 3b. PITCH CALIBRATION (optional) -------------------------------------
// Why this screen exists: "who touched the ball" is decided by a PIXEL distance
// (contact_max_player_dist_px = 80). A pixel is ~11cm of pitch near the camera
// but ~62cm at the far touchline, so the same rule allows ~8m near and ~50m far,
// and a player 50m away can be tagged with your touch. Marking the pitch lets
// that distance be judged in metres instead. See docs/detection-issues.md #14.
//
// The fit is a CAMERA (position, height, pan, tilt, roll, focal length), not a
// free 8-number transform: most of a free transform's parameter space describes
// pitches folded through the camera or shaped like bow-ties, and with sloppy
// clicks a free solver lands there while still reporting a tiny error. A camera
// cannot express those shapes at all.
//
// This mirrors polyfut_v2/pipeline/pitch_calibration.py. It exists in the
// browser for the live overlay only; the clicks are re-fitted server-side so
// the pipeline runs on a calibration made by the code that consumes it.

const PF_PITCH_LANDMARKS = [
    ['corner_near_left', '0', '0', 'Corner: left goal line x near touchline'],
    ['corner_far_left', '0', 'W', 'Corner: left goal line x far touchline'],
    ['corner_near_right', 'L', '0', 'Corner: right goal line x near touchline'],
    ['corner_far_right', 'L', 'W', 'Corner: right goal line x far touchline'],
    ['penarea_L_goalline_near', '0', 'W/2-20.16', 'Left penalty area, ON goal line, near side'],
    ['penarea_L_goalline_far', '0', 'W/2+20.16', 'Left penalty area, ON goal line, far side'],
    ['penarea_L_outer_near', '16.5', 'W/2-20.16', 'Left penalty area, OUTER corner, near side'],
    ['penarea_L_outer_far', '16.5', 'W/2+20.16', 'Left penalty area, OUTER corner, far side'],
    ['goalarea_L_goalline_near', '0', 'W/2-9.16', 'Left 6-yard box, ON goal line, near side'],
    ['goalarea_L_goalline_far', '0', 'W/2+9.16', 'Left 6-yard box, ON goal line, far side'],
    ['goalarea_L_outer_near', '5.5', 'W/2-9.16', 'Left 6-yard box, OUTER corner, near side'],
    ['goalarea_L_outer_far', '5.5', 'W/2+9.16', 'Left 6-yard box, OUTER corner, far side'],
    ['penspot_L', '11', 'W/2', 'Left penalty spot'],
    ['post_L_near', '0', 'W/2-3.66', 'Left goal: near post BASE'],
    ['post_L_far', '0', 'W/2+3.66', 'Left goal: far post BASE'],
    ['halfway_near', 'L/2', '0', 'Halfway line meets NEAR touchline'],
    ['halfway_far', 'L/2', 'W', 'Halfway line meets FAR touchline'],
    ['centre_spot', 'L/2', 'W/2', 'Centre spot'],
    ['circle_near', 'L/2', 'W/2-9.15', 'Centre circle, NEAR-side extreme'],
    ['circle_far', 'L/2', 'W/2+9.15', 'Centre circle, FAR-side extreme'],
    ['circle_left', 'L/2-9.15', 'W/2', 'Centre circle, LEFT extreme'],
    ['circle_right', 'L/2+9.15', 'W/2', 'Centre circle, RIGHT extreme'],
    ['penarea_R_goalline_near', 'L', 'W/2-20.16', 'Right penalty area, ON goal line, near side'],
    ['penarea_R_goalline_far', 'L', 'W/2+20.16', 'Right penalty area, ON goal line, far side'],
    ['penarea_R_outer_near', 'L-16.5', 'W/2-20.16', 'Right penalty area, OUTER corner, near side'],
    ['penarea_R_outer_far', 'L-16.5', 'W/2+20.16', 'Right penalty area, OUTER corner, far side'],
    ['goalarea_R_goalline_near', 'L', 'W/2-9.16', 'Right 6-yard box, ON goal line, near side'],
    ['goalarea_R_goalline_far', 'L', 'W/2+9.16', 'Right 6-yard box, ON goal line, far side'],
    ['goalarea_R_outer_near', 'L-5.5', 'W/2-9.16', 'Right 6-yard box, OUTER corner, near side'],
    ['goalarea_R_outer_far', 'L-5.5', 'W/2+9.16', 'Right 6-yard box, OUTER corner, far side'],
    ['penspot_R', 'L-11', 'W/2', 'Right penalty spot'],
    ['post_R_near', 'L', 'W/2-3.66', 'Right goal: near post BASE'],
    ['post_R_far', 'L', 'W/2+3.66', 'Right goal: far post BASE'],
];
const PF_PITCH_L = 100, PF_PITCH_W = 64;

let pfPitchFrames = [];
let pfPitchCur = 0;
let pfPitchClicks = [];
let pfPitchFit = null;
let pfPitchDone = null;          // callback: (calibrationPayload | null)

function pfLandmarkXY(key) {
    const d = PF_PITCH_LANDMARKS.find(function (l) { return l[0] === key; });
    if (!d) return null;
    try {
        const ev = function (s) {
            return Function('L', 'W', 'return (' + s + ')')(PF_PITCH_L, PF_PITCH_W);
        };
        return [ev(d[1]), ev(d[2])];
    } catch (e) { return null; }
}

// Camera axes: x=right, y=down, z=forward, so y = z x x. At pan=0 (looking along
// +X with world up = +Z) right must be -Y and down must be -Z. Getting these
// signs wrong flips the image, and a test that both generates and fits with the
// same convention cannot catch it, so they are pinned rather than assumed.
function pfRotation(pan, tilt, roll) {
    const fwd = [Math.cos(tilt) * Math.cos(pan), Math.cos(tilt) * Math.sin(pan), -Math.sin(tilt)];
    let right = [Math.sin(pan), -Math.cos(pan), 0];
    let down = [fwd[1] * right[2] - fwd[2] * right[1],
                fwd[2] * right[0] - fwd[0] * right[2],
                fwd[0] * right[1] - fwd[1] * right[0]];
    const n = Math.hypot(down[0], down[1], down[2]);
    if (n < 1e-12) return null;
    down = down.map(function (v) { return v / n; });
    if (roll) {
        const c = Math.cos(roll), s = Math.sin(roll);
        const r2 = right.map(function (v, i) { return c * v + s * down[i]; });
        const d2 = right.map(function (v, i) { return -s * v + c * down[i]; });
        right = r2; down = d2;
    }
    return [right, down, fwd];
}

function pfPoseH(p, cx, cy) {
    const R = pfRotation(p[3], p[4], p[6]);
    if (!R) return null;
    const C = [p[0], p[1], p[2]], f = p[5];
    const t = R.map(function (r) { return -(r[0] * C[0] + r[1] * C[1] + r[2] * C[2]); });
    const M = [[R[0][0], R[0][1], t[0]], [R[1][0], R[1][1], t[1]], [R[2][0], R[2][1], t[2]]];
    return [f * M[0][0] + cx * M[2][0], f * M[0][1] + cx * M[2][1], f * M[0][2] + cx * M[2][2],
            f * M[1][0] + cy * M[2][0], f * M[1][1] + cy * M[2][1], f * M[1][2] + cy * M[2][2],
            M[2][0], M[2][1], M[2][2]];
}

function pfApplyH(H, q) {
    const w = H[6] * q[0] + H[7] * q[1] + H[8];
    if (!(Math.abs(w) > 1e-9)) return null;
    return [(H[0] * q[0] + H[1] * q[1] + H[2]) / w, (H[3] * q[0] + H[4] * q[1] + H[5]) / w];
}

function pfInv3(H) {
    const a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7], i = H[8];
    const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (!isFinite(det) || Math.abs(det) < 1e-14) return null;
    return [A / det, (c * h - b * i) / det, (b * f - c * e) / det,
            B / det, (a * i - c * g) / det, (c * d - a * f) / det,
            C / det, (b * g - a * h) / det, (a * e - b * d) / det];
}

function pfSolveLS(A, b, n) {
    const M = A.map(function (r, i) { return r.concat([b[i]]); });
    for (let c = 0; c < n; c++) {
        let piv = c;
        for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
        if (Math.abs(M[piv][c]) < 1e-12) return null;
        const tmp = M[c]; M[c] = M[piv]; M[piv] = tmp;
        for (let r = 0; r < n; r++) {
            if (r === c) continue;
            const k = M[r][c] / M[c][c];
            for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j];
        }
    }
    return M.map(function (r, i) { return r[n] / M[i][i]; });
}

// Parameters: [Xc, Yc, h, f, roll, pan_0, tilt_0, pan_1, tilt_1, ...]
// Position, height, focal length and roll are SHARED across frames because a
// tripod only rotates; only pan/tilt vary. Each click is fitted in its OWN
// frame's pixels — mapping clicks into a common reference frame first would run
// them through the camera-motion chain, which measurably corrupts the fit.
function pfPoseFor(v, k) {
    return [v[0], v[1], v[2], v[5 + 2 * k], v[6 + 2 * k], v[3], v[4]];
}

function pfMultiResid(v, groups, cx, cy) {
    const out = [];
    for (let k = 0; k < groups.length; k++) {
        const g = groups[k];
        const H = pfPoseH(pfPoseFor(v, k), cx, cy);
        for (let i = 0; i < g.pit.length; i++) {
            if (!H) { out.push(1e4, 1e4); continue; }
            const q = g.pit[i];
            const w = H[6] * q[0] + H[7] * q[1] + H[8];
            if (!(w > 1e-9)) { out.push(1e4, 1e4); continue; }
            out.push((H[0] * q[0] + H[1] * q[1] + H[2]) / w - g.img[i][0],
                     (H[3] * q[0] + H[4] * q[1] + H[5]) / w - g.img[i][1]);
        }
    }
    return out;
}

function pfLM(p0, groups, cx, cy, lo, hi, iters) {
    let p = p0.slice();
    const n = p.length;
    let lam = 1e-3;
    const cost = function (v) { return v.reduce(function (a, b) { return a + b * b; }, 0); };
    let r = pfMultiResid(p, groups, cx, cy), c = cost(r);
    for (let it = 0; it < iters; it++) {
        const m = r.length, J = [];
        for (let k = 0; k < n; k++) {
            const step = Math.max(1e-6, Math.abs(p[k]) * 1e-6);
            const pp = p.slice();
            pp[k] = Math.min(hi[k], Math.max(lo[k], pp[k] + step));
            const rr = pfMultiResid(pp, groups, cx, cy);
            const dd = (pp[k] - p[k]) || step;
            J.push(rr.map(function (v, i) { return (v - r[i]) / dd; }));
        }
        const A = [], g = [];
        for (let a = 0; a < n; a++) {
            const row = [];
            for (let b = 0; b < n; b++) {
                let s = 0; for (let i = 0; i < m; i++) s += J[a][i] * J[b][i];
                row.push(s);
            }
            A.push(row);
            let s2 = 0; for (let i = 0; i < m; i++) s2 += J[a][i] * r[i];
            g.push(-s2);
        }
        let improved = false;
        for (let att = 0; att < 6 && !improved; att++) {
            const M = A.map(function (row, i) {
                return row.map(function (v, j) { return i === j ? v * (1 + lam) : v; });
            });
            const dp = pfSolveLS(M, g, n);
            if (!dp || dp.some(function (v) { return !isFinite(v); })) { lam *= 10; continue; }
            const pn = p.map(function (v, i) { return Math.min(hi[i], Math.max(lo[i], v + dp[i])); });
            const rn = pfMultiResid(pn, groups, cx, cy), cn = cost(rn);
            if (cn < c) { p = pn; r = rn; c = cn; lam = Math.max(lam * 0.3, 1e-9); improved = true; }
            else lam *= 10;
        }
        if (!improved) break;
    }
    return p;
}

function pfPitchGroups() {
    const byFrame = new Map();
    pfPitchClicks.forEach(function (p) {
        const pit = pfLandmarkXY(p.landmark);
        if (!pit) return;
        if (!byFrame.has(p.frame_index)) {
            byFrame.set(p.frame_index, { frame: p.frame_index, img: [], pit: [] });
        }
        const g = byFrame.get(p.frame_index);
        g.img.push([p.x, p.y]);      // RAW native-video pixels, no registration
        g.pit.push(pit);
    });
    return Array.from(byFrame.values());
}

function pfFitPitch() {
    const groups = pfPitchGroups();
    const total = groups.reduce(function (a, g) { return a + g.pit.length; }, 0);
    if (total < 4 || !pfPitchFrames.length) return null;
    const fr = pfPitchFrames[0];
    const cx = fr.width / 2, cy = fr.height / 2;
    const nf = groups.length;
    const lo = [-40, -60, 1.2, 150, -12 * Math.PI / 180];
    const hi = [PF_PITCH_L + 40, -0.5, 45, 6000, 12 * Math.PI / 180];
    for (let k = 0; k < nf; k++) {
        lo.push(-Math.PI, 1 * Math.PI / 180);
        hi.push(Math.PI, 85 * Math.PI / 180);
    }
    // Pan must be swept: the optimiser cannot rotate the camera halfway round
    // the pitch by itself. Tilt and focal length are far less multi-modal.
    let best = null;
    const pans = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180];
    for (let a = 0; a < pans.length; a++) {
        for (const tilt of [10, 28]) {
            const seed = [PF_PITCH_L / 2, -12, 8, 800, 0];
            for (let k = 0; k < nf; k++) seed.push(pans[a] * Math.PI / 180, tilt * Math.PI / 180);
            for (let i = 0; i < seed.length; i++) seed[i] = Math.min(hi[i], Math.max(lo[i], seed[i]));
            const p = pfLM(seed, groups, cx, cy, lo, hi, 35);
            const rr = pfMultiResid(p, groups, cx, cy);
            const d = [];
            for (let i = 0; i < rr.length; i += 2) d.push(Math.hypot(rr[i], rr[i + 1]));
            if (!d.length || d.some(function (v) { return !isFinite(v) || v > 4000; })) continue;
            const sorted = d.slice().sort(function (x, y) { return x - y; });
            const med = sorted[Math.floor(sorted.length / 2)];
            // median, not mean: one mislabelled landmark must not pick the winner
            if (!best || med < best.med) {
                best = { p: p, med: med, groups: groups, cx: cx, cy: cy,
                         rms: Math.sqrt(d.reduce(function (x, y) { return x + y * y; }, 0) / d.length) };
            }
        }
    }
    return best;
}

function pfPitchModel(L, W) {
    const segs = [[[0, 0], [L, 0]], [[0, W], [L, W]], [[0, 0], [0, W]],
                  [[L, 0], [L, W]], [[L / 2, 0], [L / 2, W]]];
    [[0, 1], [L, -1]].forEach(function (e) {
        const gx = e[0], s = e[1];
        [[16.5, 20.16], [5.5, 9.16]].forEach(function (b) {
            const dep = b[0], half = b[1];
            segs.push([[gx, W / 2 - half], [gx + s * dep, W / 2 - half]],
                      [[gx + s * dep, W / 2 - half], [gx + s * dep, W / 2 + half]],
                      [[gx + s * dep, W / 2 + half], [gx, W / 2 + half]]);
        });
    });
    const arcs = [];
    const circ = function (cx, cy, r, a0, a1) {
        const pts = [];
        for (let k = 0; k <= 48; k++) {
            const th = (a0 + (a1 - a0) * k / 48) * Math.PI / 180;
            pts.push([cx + r * Math.cos(th), cy + r * Math.sin(th)]);
        }
        arcs.push(pts);
    };
    circ(L / 2, W / 2, 9.15, 0, 360);
    circ(11, W / 2, 9.15, -53, 53);
    circ(L - 11, W / 2, 9.15, 127, 233);
    return { segs: segs, arcs: arcs };
}

// The decisive check. A calibration can have tiny residuals AND be insensitive
// to click noise while still being globally wrong; only seeing the model land on
// the real painted lines rules that out.
// Plane-to-plane transform straight from the anchors: pitch metres -> this
// frame's pixels. Normalised DLT, least squares. With four anchors it passes
// through them exactly; with more it is the closest single projection to all of
// them. Nothing about cameras, pitch dimensions or lens behaviour enters here,
// which is the point — those are the assumptions that were fighting the user.
function pfHomographyFrom(src, dst) {
    const n = src.length;
    if (n < 4) return null;
    const norm = function (pts) {
        let cx = 0, cy = 0;
        pts.forEach(function (p) { cx += p[0]; cy += p[1]; });
        cx /= pts.length; cy /= pts.length;
        let d = 0;
        pts.forEach(function (p) { d += Math.hypot(p[0] - cx, p[1] - cy); });
        d /= pts.length;
        if (!(d > 1e-9)) return null;
        const sc = Math.SQRT2 / d;
        return {sc: sc, cx: cx, cy: cy,
                out: pts.map(function (p) { return [(p[0]-cx)*sc, (p[1]-cy)*sc]; })};
    };
    const S = norm(src), D = norm(dst);
    if (!S || !D) return null;
    const AtA = [], Atb = new Array(8).fill(0);
    for (let i = 0; i < 8; i++) AtA.push(new Array(8).fill(0));
    for (let i = 0; i < n; i++) {
        const x = S.out[i][0], y = S.out[i][1];
        const u = D.out[i][0], v = D.out[i][1];
        const rows = [[x, y, 1, 0, 0, 0, -u*x, -u*y, u],
                      [0, 0, 0, x, y, 1, -v*x, -v*y, v]];
        for (const r of rows) {
            for (let a = 0; a < 8; a++) {
                for (let b = 0; b < 8; b++) AtA[a][b] += r[a] * r[b];
                Atb[a] += r[a] * r[8];
            }
        }
    }
    const h = pfSolveLS(AtA, Atb, 8);
    if (!h || h.some(function (v) { return !isFinite(v); })) return null;
    const Hn = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
    const Ts = [S.sc, 0, -S.sc*S.cx, 0, S.sc, -S.sc*S.cy, 0, 0, 1];
    const Ti = [1/D.sc, 0, D.cx, 0, 1/D.sc, D.cy, 0, 0, 1];
    const mul = function (X, Y) {
        const O = new Array(9).fill(0);
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
            for (let k = 0; k < 3; k++) O[i*3+j] += X[i*3+k] * Y[k*3+j];
        return O;
    };
    let H = mul(Ti, mul(Hn, Ts));
    // Fix the sign so that w > 0 means "in front of the camera". H and -H are
    // the same projective transform, so a solver may hand back either, and
    // without a convention there is no way to tell an in-front point from a
    // behind-the-camera one. The anchors are all visibly in frame, so whichever
    // sign makes THEM positive is the right one.
    let votes = 0;
    for (let i = 0; i < n; i++) {
        const w = H[6]*src[i][0] + H[7]*src[i][1] + H[8];
        votes += (w < 0) ? 1 : -1;
    }
    if (votes > 0) H = H.map(function (v) { return -v; });
    return H;
}

// Where a pitch segment crosses the horizon, i.e. where w hits zero. Beyond it
// the projection flips to the opposite side of the image, which is what drew
// the pitch wrapping back across the frame.
const PF_W_EPS = 1e-6;
// A point clipped to just in front of the horizon projects enormously far away
// (measured: 6.3e8 px at the epsilon above). The direction is correct and the
// canvas would clip it, but coordinates that size are asking for precision
// trouble, so the far end is pulled back along the same ray to something sane.
const PF_MAX_REACH = 20000;

function pfClipToFront(H, a, b) {
    const wOf = function (q) { return H[6]*q[0] + H[7]*q[1] + H[8]; };
    const wa = wOf(a), wb = wOf(b);
    if (wa <= PF_W_EPS && wb <= PF_W_EPS) return null;    // wholly behind
    let A = a, B = b;
    if (wa <= PF_W_EPS || wb <= PF_W_EPS) {
        // Walk the crossing point to just in front of the horizon. Drawing to
        // the exact crossing sends the line to infinity, so stop short of it.
        const t = (wa - PF_W_EPS) / (wa - wb);
        const mid = [a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t];
        if (wa > PF_W_EPS) B = mid; else A = mid;
    }
    let P = pfApplyH(H, A), Q = pfApplyH(H, B);
    if (!P || !Q) return null;
    // Keep the ray, bound the length.
    const shorten = function (from, to) {
        const dx = to[0] - from[0], dy = to[1] - from[1];
        const d = Math.hypot(dx, dy);
        if (!(d > PF_MAX_REACH)) return to;
        const s = PF_MAX_REACH / d;
        return [from[0] + dx * s, from[1] + dy * s];
    };
    if (Math.hypot(P[0], P[1]) < Math.hypot(Q[0], Q[1])) Q = shorten(P, Q);
    else P = shorten(Q, P);
    if (!isFinite(P[0]) || !isFinite(P[1]) || !isFinite(Q[0]) || !isFinite(Q[1]))
        return null;
    return [P, Q];
}

// The transform used for THIS frame: anchors first, camera model only as a
// fallback for a frame the user has not marked up.
function pfFrameH(frameIndex) {
    if (!pfPitchFit) return null;
    const k = pfPitchFit.groups.findIndex(function (g) { return g.frame === frameIndex; });
    if (k < 0) return null;
    const g = pfPitchFit.groups[k];
    const direct = pfHomographyFrom(g.pit, g.img);
    if (direct) return direct;
    return pfPoseH(pfPoseFor(pfPitchFit.p, k), pfPitchFit.cx, pfPitchFit.cy);
}

function pfDrawPitchOverlay() {
    const cv = document.getElementById('cv-pitch-canvas');
    const img = document.getElementById('cv-pitch-img');
    if (!cv || !img || !pfPitchFrames.length) return;
    const fr = pfPitchFrames[pfPitchCur];
    cv.width = fr.width; cv.height = fr.height;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!pfPitchFit) return;
    const H = pfFrameH(fr.frame_index);
    if (!H) return;      // no anchors on this frame yet — nothing to draw
    const model = pfPitchModel(PF_PITCH_L, PF_PITCH_W);
    ctx.strokeStyle = '#ffe14d';
    ctx.lineWidth = Math.max(2, fr.width / 480);
    ctx.globalAlpha = 0.9;
    model.segs.forEach(function (sg) {
        const seg = pfClipToFront(H, sg[0], sg[1]);
        if (!seg) return;
        ctx.beginPath();
        ctx.moveTo(seg[0][0], seg[0][1]);
        ctx.lineTo(seg[1][0], seg[1][1]);
        ctx.stroke();
    });
    model.arcs.forEach(function (pl) {
        ctx.beginPath();
        let started = false;
        for (let i = 0; i + 1 < pl.length; i++) {
            const seg = pfClipToFront(H, pl[i], pl[i + 1]);
            if (!seg) { started = false; continue; }
            if (!started) { ctx.moveTo(seg[0][0], seg[0][1]); started = true; }
            else { ctx.lineTo(seg[0][0], seg[0][1]); }
            ctx.lineTo(seg[1][0], seg[1][1]);
        }
        ctx.stroke();
    });
}

// Per-mark reprojection error, so a bad mark can be named instead of the user
// being told only that one exists somewhere.
function pfMarkErrors() {
    const out = new Map();
    if (!pfPitchFit) return out;
    pfPitchFit.groups.forEach(function (g, k) {
        // the transform actually drawn, so a number shown next to a mark means
        // "how far the drawn pitch misses it" rather than "how far an idealised
        // camera would have missed it"
        const H = pfFrameH(g.frame);
        if (!H) return;
        g.pit.forEach(function (q, i) {
            const w = H[6] * q[0] + H[7] * q[1] + H[8];
            if (!(w > 1e-9)) { out.set(g.frame + ':' + i, Infinity); return; }
            const x = (H[0] * q[0] + H[1] * q[1] + H[2]) / w;
            const y = (H[3] * q[0] + H[4] * q[1] + H[5]) / w;
            out.set(g.frame + ':' + i,
                    Math.hypot(x - g.img[i][0], y - g.img[i][1]));
        });
    });
    return out;
}

// Error for each entry of pfPitchClicks, in click order.
function pfClickErrors() {
    const per = pfMarkErrors();
    const seen = new Map();
    return pfPitchClicks.map(function (p) {
        const n = seen.get(p.frame_index) || 0;
        seen.set(p.frame_index, n + 1);
        const e = per.get(p.frame_index + ':' + n);
        return (e === undefined) ? null : e;
    });
}

function pfPitchQuality() {
    const el = document.getElementById('cv-pitch-quality');
    const useBtn = document.getElementById('cv-pitch-use');
    if (!el) return;
    const distinct = new Set(pfPitchClicks.map(function (p) { return p.landmark; }));
    // A duplicate is the same landmark twice ON THE SAME FRAME. The same
    // landmark on two different frames is legitimate — that is what lets one
    // camera be fitted across a pan — and flagging it blocked real work.
    const perFrame = new Set();
    const dupes = [];
    pfPitchClicks.forEach(function (p) {
        const key = p.frame_index + '|' + p.landmark;
        if (perFrame.has(key)) {
            if (dupes.indexOf(p.landmark) < 0) dupes.push(p.landmark);
        }
        perFrame.add(key);
    });
    const msgs = [];
    let cls = '', ok = false;

    if (dupes.length) {
        msgs.push('<b>' + dupes.join(', ') + ' is marked twice on the same frame.</b> ' +
                  'A landmark is one spot on the pitch, so two positions for it on ' +
                  'the same frame contradict each other. Tap the wrong one in the ' +
                  'list to remove it.');
        cls = 'is-bad';
    }
    if (distinct.size < 4) {
        msgs.push('<b>' + distinct.size + ' of 4 landmarks.</b> Four different ones are ' +
                  'the minimum; below that there are endless possible answers and ' +
                  'the app would pick a wrong one without any sign.');
        el.className = 'cv-pitch-quality is-bad';
        el.innerHTML = msgs.join('<br><br>');
        if (useBtn) useBtn.disabled = true;
        return;
    }

    if (pfPitchFit) {
        const p = pfPitchFit.p;
        const nPar = 5 + 2 * pfPitchFit.groups.length;
        const dof = 2 * distinct.size - nPar;
        const panDeg = p[5] * 180 / Math.PI;
        const across = Math.abs(Math.abs(panDeg) - 90) < 25;
        msgs.push('<b>This says your camera was:</b> ' + p[2].toFixed(1) + ' m high, ' +
                  Math.abs(p[1]).toFixed(0) + ' m back from the touchline, pointed ' +
                  (across ? 'across the pitch' : panDeg.toFixed(0) + '&deg;') +
                  ', tilted ' + (p[6] * 180 / Math.PI).toFixed(0) + '&deg; down.<br>' +
                  '<i>A rough sanity check on where you filmed from — the pitch ' +
                  'above is drawn from your marks either way.</i>');
        // Error against the DRAWN transform. With four anchors on a frame this
        // is ~0 by construction; a large value means the anchors on one frame
        // disagree with each other, which is a real contradiction rather than an
        // assumption of ours.
        const drawnErrs = pfClickErrors().filter(function (e) { return e !== null; });
        drawnErrs.sort(function (a, b) { return a - b; });
        const drawnMed = drawnErrs.length
            ? drawnErrs[Math.floor(drawnErrs.length / 2)] : null;
        if (drawnMed !== null && drawnMed <= 4) {
            msgs.push('<b style="color:#2ecc71">The pitch is drawn through your ' +
                      'marks</b> (within ' + drawnMed.toFixed(1) + ' px). If the ' +
                      'yellow lines still do not sit on the painted ones, add a ' +
                      'mark where they drift.');
            if (!cls) cls = 'is-good';
            ok = true;
        } else if (drawnMed !== null) {
            msgs.push('<b style="color:#ffd166">The pitch is drawn as close to ' +
                      'your marks as one flat surface allows</b> (about ' +
                      drawnMed.toFixed(0) + ' px off). Usually this means the ' +
                      'penalty boxes or circle on this ground are not regulation ' +
                      'size, so the box marks and the touchline marks pull in ' +
                      'different directions. It is still usable. If one mark is ' +
                      'much worse than the rest, that one is worth a second look.');
            if (cls !== 'is-bad') cls = 'is-warn';
            ok = true;
        }
        if (false) {
            // Name the offender rather than leaving the user to hunt for it.
            const errs = pfClickErrors();
            let wi = -1, wv = -1;
            errs.forEach(function (e, i) { if (e !== null && e > wv) { wv = e; wi = i; } });
            const who = (wi >= 0 && wv > 8)
                ? '<span class=mut>The drawn pitch misses <b>' +
                  pfPitchClicks[wi].landmark + '</b> at ' +
                  pfPitchClicks[wi].t_sec.toFixed(0) + 's by ' + wv.toFixed(0) +
                  ' px — if that mark looks right where it is, the two around it ' +
                  'are worth a second look. Tap any mark to remove it.</span>'
                : '';
            msgs.push('' + who);
        }
        // The rigid-camera fit is only a note now. It assumes a regulation
        // 100x64 pitch with regulation box sizes, no lens distortion and the
        // lens dead centre — none of which a community pitch owes us — so its
        // disagreement says more about those assumptions than about the marks.
        if (pfPitchFit.med > 10) {
            msgs.push('<span class=mut>A regulation-sized pitch does not quite ' +
                      'match these marks (' + pfPitchFit.med.toFixed(0) + ' px ' +
                      'off). That is expected if this pitch or its boxes are not ' +
                      'standard size, and the drawing above ignores it.</span>');
        }
        if (dof < 4) {
            msgs.push('<i>Only ' + dof + ' to spare (' + nPar + ' unknowns vs ' +
                      (2 * distinct.size) + ' facts). A low error does not prove much ' +
                      'yet &mdash; aim for 8 landmarks.</i>');
            if (cls === 'is-good') cls = 'is-warn';
        }
        msgs.push('<b>Check the yellow pitch sits on the real lines.</b> That is the ' +
                  'one check that cannot be fooled.');
    }
    if (dupes.length) ok = false;
    el.className = 'cv-pitch-quality ' + cls;
    el.innerHTML = msgs.join('<br><br>');
    if (useBtn) useBtn.disabled = !ok;
}

function pfRenderPitch() {
    const img = document.getElementById('cv-pitch-img');
    const dots = document.getElementById('cv-pitch-dots');
    const tabs = document.getElementById('cv-pitch-tabs');
    if (!img || !pfPitchFrames.length) return;
    const fr = pfPitchFrames[pfPitchCur];
    if (img.getAttribute('data-frame') !== String(fr.frame_index)) {
        img.src = 'data:image/jpeg;base64,' + fr.jpeg_b64;
        img.setAttribute('data-frame', String(fr.frame_index));
    }
    Array.prototype.forEach.call(tabs.children, function (b, k) {
        b.setAttribute('aria-current', k === pfPitchCur ? 'true' : 'false');
    });
    dots.innerHTML = '';
    pfPitchClicks.filter(function (p) { return p.frame_index === fr.frame_index; })
        .forEach(function (p) {
            const d = document.createElement('div');
            d.className = 'cv-pitch-dot';
            d.style.left = (p.x / fr.width * 100) + '%';
            d.style.top = (p.y / fr.height * 100) + '%';
            d.innerHTML = '<b>' + p.landmark + '</b>';
            dots.appendChild(d);
        });
    pfPitchFit = pfFitPitch();
    pfDrawPitchOverlay();
    pfPitchQuality();
    const list = document.getElementById('cv-pitch-list');
    if (list) {
        const errs = pfClickErrors();
        list.innerHTML = pfPitchClicks.map(function (p, i) {
            const e = errs[i];
            const bad = (e !== null && e > 8);
            return '<tr class="cv-pitch-row' + (bad ? ' is-bad' : '') +
                   '" data-i="' + i + '" title="Tap to remove this mark">' +
                   '<td>' + p.landmark + '</td><td>' + p.t_sec.toFixed(0) + 's</td>' +
                   '<td>' + (e === null ? '' : e.toFixed(0) + 'px') + '</td></tr>';
        }).join('');
        Array.prototype.forEach.call(list.querySelectorAll('tr'), function (tr) {
            tr.onclick = function () {
                const i = parseInt(tr.getAttribute('data-i'), 10);
                if (!isNaN(i)) { pfPitchClicks.splice(i, 1); pfRenderPitch(); }
            };
        });
    }
}

function pfPitchPayload() {
    if (!pfPitchFit || !pfPitchFrames.length) return null;
    const fr = pfPitchFrames[0];
    return {
        clicks: pfPitchClicks.map(function (p) {
            return { frame_index: p.frame_index, x: p.x, y: p.y, landmark: p.landmark };
        }),
        // the browser's own answer, sent as a starting point for the server's
        // authoritative re-fit (verified there, not trusted)
        params: pfPitchFit.p,
        frame_width: fr.width,
        frame_height: fr.height,
        pitch_length_m: PF_PITCH_L,
        pitch_width_m: PF_PITCH_W,
    };
}

function showPitchCalibScreen(token, playRanges, done) {
    pfPitchDone = done;
    pfPitchClicks = [];
    pfPitchFit = null;
    pfPitchCur = 0;
    const screen = document.getElementById('cv-pitch-screen');
    const sel = document.getElementById('cv-pitch-landmark');
    if (!screen || !sel) { done(null); return; }
    screen.classList.remove('hidden');

    if (!sel.options.length) {
        PF_PITCH_LANDMARKS.forEach(function (l) {
            const o = document.createElement('option');
            o.value = l[0];
            o.textContent = l[3];
            sel.appendChild(o);
        });
    }

    const fd = new FormData();
    fd.append('token', token);
    fd.append('count', '6');
    fd.append('play_ranges', JSON.stringify(playRanges || []));
    fetch(cvApiUrl('/api/v2/calibration_frames'), { method: 'POST', body: fd })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (!data || data.error || !(data.frames || []).length) throw new Error('no frames');
            pfPitchFrames = data.frames;
            const tabs = document.getElementById('cv-pitch-tabs');
            tabs.innerHTML = '';
            pfPitchFrames.forEach(function (f, k) {
                const b = document.createElement('button');
                b.type = 'button';
                b.textContent = f.t_sec.toFixed(0) + 's';
                b.onclick = function () { pfPitchCur = k; pfRenderPitch(); };
                tabs.appendChild(b);
            });
            pfRenderPitch();
        })
        .catch(function () {
            // Calibration is an enhancement; if frames can't be fetched just
            // carry on with today's behaviour rather than blocking the user.
            pfFinishPitch(null);
        });
}

function pfFinishPitch(payload) {
    const screen = document.getElementById('cv-pitch-screen');
    if (screen) screen.classList.add('hidden');
    const cb = pfPitchDone;
    pfPitchDone = null;
    if (cb) cb(payload);
}

function pfInitPitchScreen() {
    const img = document.getElementById('cv-pitch-img');
    const sel = document.getElementById('cv-pitch-landmark');
    if (!img || img.getAttribute('data-wired')) return;
    img.setAttribute('data-wired', '1');

    img.addEventListener('click', function (e) {
        if (!pfPitchFrames.length) return;
        const fr = pfPitchFrames[pfPitchCur];
        const r = img.getBoundingClientRect();
        // clicks recorded in NATIVE video pixels, whatever size it is displayed at
        const x = (e.clientX - r.left) / r.width * fr.width;
        const y = (e.clientY - r.top) / r.height * fr.height;
        pfPitchClicks.push({
            landmark: sel.value, frame_index: fr.frame_index,
            t_sec: fr.t_sec, x: x, y: y,
        });
        // Deliberately NOT auto-advancing the dropdown. It silently relabels:
        // click twice without watching it and the second mark is recorded as a
        // landmark you never chose, which is indistinguishable from a bad click.
        pfRenderPitch();
    });

    const undo = document.getElementById('cv-pitch-undo');
    if (undo) undo.onclick = function () { pfPitchClicks.pop(); pfRenderPitch(); };
    const clear = document.getElementById('cv-pitch-clear');
    if (clear) clear.onclick = function () { pfPitchClicks = []; pfRenderPitch(); };
    const skip = document.getElementById('cv-pitch-skip');
    if (skip) skip.onclick = function () { pfFinishPitch(null); };
    const use = document.getElementById('cv-pitch-use');
    if (use) use.onclick = function () { pfFinishPitch(pfPitchPayload()); };
}

document.addEventListener('DOMContentLoaded', pfInitPitchScreen);
