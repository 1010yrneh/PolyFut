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
let cvPollTimer = null;
let cvSegmentsAreDemo = false;
const CV_SERVER_PORTS = [5000, 5050, 8080];
const CV_SESSION_KEY = 'polyfut_cv_session';
const CV_CATALOGUE_KEY = 'polyfut_match_catalogue';
let cvServerBase = '';

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
}

document.addEventListener('DOMContentLoaded', () => {
    const vInput = document.getElementById('video-input');
    if (vInput) vInput.addEventListener('change', checkStartReady);
});

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
    var v2Toggle = document.getElementById('cv-v2-mode');
    cvV2Mode = !!(v2Toggle && v2Toggle.checked);
    clearReviewSession();
    captureSetupMetadataToSession();

    const setupScreen = document.getElementById('setup-screen');
    if (setupScreen) setupScreen.classList.add('hidden');

    startTeamDetection();
}

// --- 4a. TEAM COLOUR DETECTION + PICKER ---
let cvToken = null;
let cvTeams = [];
let cvMyTeam = null;
let cvMyTeamId = 'team_a';

function startTeamDetection() {
    const screen = document.getElementById('cv-team-screen');
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
    screen.classList.remove('hidden');
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
                // v2: start compiling the soccer model now so the first seed clip
                // is fast by the time the user finishes picking their team.
                if (cvV2Mode && cvToken) {
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
            sub.innerText = 'We detected these kit colours from your video. Tap the one your team wears.';
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
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'cv-team-swatch';
        const sw = document.createElement('span');
        sw.className = 'cv-team-chip';
        sw.style.backgroundColor = team.hex || '#888';
        const label = document.createElement('span');
        label.className = 'cv-team-label';
        label.innerText = team.label || (team.id === 'team_b' ? 'Team B' : 'Team A');
        btn.appendChild(sw);
        btn.appendChild(label);
        btn.onclick = function () { pickTeam(i); };
        opts.appendChild(btn);
    });
    if (!cvTeams.length && sub) sub.innerText = 'No players detected to read colours from. Try another clip.';
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
    if (cvV2Mode) {
        // v2: identify YOU (not just the team) via a few tapped frames.
        var seedVid = document.getElementById('cv-seed-video');
        var dur = (seedVid && isFinite(seedVid.duration) && seedVid.duration > 0)
            ? seedVid.duration : 0;
        showV2SeedScreen(cvVideoURL, dur);
        return;
    }
    startCvAnalysis();
}

// --- 4a. v2 SEED SCREEN: tap yourself on a tracked node in a 3s enhanced clip ---
// Each detected player gets a marker that follows them; tapping one turns that
// player's whole-clip track into seed taps [{t_sec, nx, ny}] for /api/v2/process.
let cvV2Mode = false;
let __v2Seed = null;
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

// True if a tracklet is CLEARLY on another team (hide it). Unknown colour or an
// unreliable team read → false (keep shown), per the safety choice.
function __v2SeedOtherTeam(tr) {
    if (!cvMyTeam || !cvMyTeam.hex || __v2SeedDefaultKits()) return false;
    var others = (cvTeams || []).filter(function (t) { return t !== cvMyTeam && t.hex; });
    if (!others.length) return false;
    var kh = tr && tr.kit_hsv;
    if (!kh) return false;                              // unknown → shown
    var dMine = __hsvDist(kh, __hexToHsv(cvMyTeam.hex));
    if (dMine == null) return false;
    var dOther = Infinity;
    others.forEach(function (t) {
        var d = __hsvDist(kh, __hexToHsv(t.hex));
        if (d != null && d < dOther) dOther = d;
    });
    return dOther < dMine - 15 && dOther < 90;          // clearly nearer another kit
}

function showV2SeedScreen(url, duration) {
    __v2Seed = {
        reroll: 0, index: 0, moments: [], nMoments: 4,
        clip: null, cache: {},           // cache key `${reroll}_${index}` -> clip
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
        ? 'Find my touches · ' + n + (n === 1 ? ' clip' : ' clips')
        : 'Find my touches';
}

function __v2SeedApi(path) { return cvApiUrl(path); }

function __v2SeedLoadIndex(then) {
    const s = __v2Seed;
    if (!s || !cvToken) { if (then) then(); return; }
    fetch(__v2SeedApi('/api/v2/seed_clips_index'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: cvToken, reroll: s.reroll }),
    }).then(function (r) { return r.json(); }).then(function (data) {
        if (data && data.moments) { s.moments = data.moments; s.nMoments = data.moments.length; }
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
            (s.selections[s.reroll + '_' + i] ? ' picked' : '');
        dot.title = 'Clip ' + (i + 1);
        dot.onclick = (function (idx) { return function () { __v2SeedLoadClip(idx); }; })(i);
        el.appendChild(dot);
    }
}

function __v2SeedSetLoading(on, msg) {
    const el = document.getElementById('cv-seed-loading');
    if (!el) return;
    if (msg != null) el.innerHTML = '<div class="cv-seed-loading-msg">' + msg + '</div>';
    el.classList.toggle('hidden', !on);
    if (!on) __v2SeedStopLoadTicker();
}

// While a clip builds, show a live timer. The very first build of the session
// also warns that a one-time model preparation makes it take a bit longer.
function __v2SeedStartLoadTicker(index) {
    __v2SeedStopLoadTicker();
    const first = !__v2SeedBuilt;
    const started = Date.now();
    function render() {
        const secs = Math.floor((Date.now() - started) / 1000);
        let html = 'Enhancing clip ' + (index + 1) + '… <span style="opacity:.7">' +
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

    const key = s.reroll + '_' + index;
    const hint = document.getElementById('cv-seed-hint');
    const cached = s.cache[key];
    if (cached) { __v2SeedShowClip(cached, key); __v2SeedPrefetchNext(); return; }

    __v2SeedStartLoadTicker(index);
    if (hint) {
        hint.innerText = __v2SeedBuilt
            ? 'Preparing an enhanced clip and tracking the players…'
            : 'Getting your first clip ready — this one takes a little longer.';
    }
    fetch(__v2SeedApi('/api/v2/seed_clip'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: cvToken, index: index, reroll: s.reroll }),
    }).then(function (r) { return r.json(); }).then(function (data) {
        if (!data || data.error || !data.clip_url) {
            throw new Error((data && data.error) || 'clip failed');
        }
        __v2SeedBuilt = true;
        s.cache[key] = data;
        // Only render if the user hasn't navigated away while we loaded.
        if (s.reroll + '_' + s.index === key) { __v2SeedShowClip(data, key); }
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
    const key = s.reroll + '_' + nxt;
    if (s.cache[key]) return;
    fetch(__v2SeedApi('/api/v2/seed_clip'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: cvToken, index: nxt, reroll: s.reroll }),
    }).then(function (r) { return r.json(); }).then(function (data) {
        if (data && data.clip_url) { __v2SeedBuilt = true; s.cache[s.reroll + '_' + nxt] = data; }
    }).catch(function () {});
}

function __v2SeedShowClip(clip, key) {
    const s = __v2Seed;
    s.clip = clip;
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
    // Team filter: hide only players clearly on the other team. If that would
    // hide *everyone* (bad colour read), fall back to showing all.
    let hidden = tracks.map(function (tr) { return __v2SeedOtherTeam(tr); });
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
    // Re-highlight the marker you picked earlier in this clip, if any.
    if (already) {
        s.nodeEls.forEach(function (e) {
            e.group.classList.toggle('selected',
                e.group.dataset.trackId === String(already.trackId));
        });
    }
    __v2SeedUpdateFinishBtn();
    __v2SeedShowPlayPause(false);   // starts playing → hide indicator
    vid.src = cvApiUrl(clip.clip_url);
    vid.currentTime = 0;
    const p = vid.play();
    if (p && p.catch) p.catch(function () {});
    __v2SeedStartAnim();
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
        const t = vid.currentTime || 0;
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

function __v2SeedPickNode(tr, key) {
    const s = __v2Seed;
    if (!s) return;
    const cur = s.selections[key];
    const deselect = cur && String(cur.trackId) === String(tr.id);  // tap again → clear
    if (deselect) delete s.selections[key];
    else s.selections[key] = { trackId: tr.id, taps: tr.taps || [] };
    (s.nodeEls || []).forEach(function (e) {
        e.group.classList.toggle('selected',
            !deselect && e.group.dataset.trackId === String(tr.id));
    });
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

function __v2SeedReroll() {
    const s = __v2Seed;
    if (!s) return;
    s.reroll += 1;   // selections persist — every marked clip still counts
    __v2SeedLoadIndex(function () { __v2SeedLoadClip(0); });
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

// --- 4b. RUN ANALYSIS (token + chosen colours -> poll) ---
function startCvAnalysis() {
    const proc = document.getElementById('cv-processing-screen');
    proc.classList.remove('hidden');
    showProcessTracker();
    setCvProgress({ progress: 0, status: 'Starting analysis…', stage: 'init' });

    // No token means we're offline (demo) -> straight to browser demo.
    if (!cvToken) {
        runBrowserDemo();
        return;
    }

    const fd = new FormData();
    fd.append('token', cvToken);
    if (cvMyTeamId) {
        fd.append('my_team', cvMyTeamId);
    }
    var meta = getSetupMetadataFields();
    fd.append('opponent', meta.opponent || '');
    fd.append('match_date', meta.match_date || '');
    fd.append('score_us', String(meta.score_us != null ? meta.score_us : 0));
    fd.append('score_them', String(meta.score_them != null ? meta.score_them : 0));
    fd.append('position', meta.position || '');

    fetch(cvApiUrl('/api/process'), { method: 'POST', body: fd })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) throw new Error(data.error);
            cvJobId = data.job_id;
            cvSegmentsAreDemo = false;
            saveCvSession({
                job_id: cvJobId,
                token: cvToken,
                my_team: cvMyTeamId,
                state: 'running'
            });
            captureSetupMetadataToSession();
            if (data.resumed) {
                setCvProgress({ progress: 0, status: 'Reconnected to analysis in progress…', stage: 'inference' });
            }
            // region agent log
            __dbgJs('B5', 'script.js:startCvAnalysis', 'job started', {
                job_id: cvJobId,
                my_team: cvMyTeamId
            });
            // endregion
            pollCvStatus();
        })
        .catch(function () {
            // Server became unreachable mid-flow -> fall back to browser demo.
            runBrowserDemo();
        });
}

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
    if (cvMyTeamId) fd.append('my_team', cvMyTeamId);
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
            cvV2Mode = true;
            saveCvSession({ job_id: cvJobId, token: cvToken, my_team: cvMyTeamId, state: 'running', v2: true });
            captureSetupMetadataToSession();
            pollCvStatus();
        })
        .catch(function () { runBrowserDemo(); });
}

function enterV2Review(j) {
    hideProcessTracker();
    document.getElementById('cv-processing-screen').classList.add('hidden');
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

    var vid = document.getElementById('cv-montage-video');
    vid.src = cvApiUrl('/api/video/' + __v2Montage.token);
    vid.load();

    if (queue.length === 0) { __v2MontageFinalize(); return; }
    __v2MontageShow();
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
function __v2MontagePosAt(points, t) {
    if (!points || !points.length) return null;
    if (t <= points[0].t_sec) return points[0];
    var last = points[points.length - 1];
    if (t >= last.t_sec) return last;
    for (var i = 0; i < points.length - 1; i++) {
        var a = points[i], b = points[i + 1];
        if (t >= a.t_sec && t <= b.t_sec) {
            var f = (b.t_sec - a.t_sec) > 1e-6 ? (t - a.t_sec) / (b.t_sec - a.t_sec) : 0;
            return { nx: a.nx + (b.nx - a.nx) * f, ny: a.ny + (b.ny - a.ny) * f };
        }
    }
    return last;
}

// Fetch (and cache on the item) the tracklet that makes the ring follow the
// reviewed player. Falls back silently to the fixed crop on any failure.
function __v2MontageFetchTrack(it) {
    var m = __v2Montage;
    if (!m || !it || it.__track !== undefined) return;
    it.__track = null;   // "requested" — avoid duplicate fetches
    fetch(cvApiUrl('/api/v2/review_track/' + m.jobId + '/' + it.rank))
        .then(function (r) { return r.json(); })
        .then(function (d) { it.__track = (d && d.points) ? d.points : []; })
        .catch(function () { it.__track = []; });
}

function __v2MontageShow() {
    var m = __v2Montage; if (!m) return;
    var it = m.queue[0];
    if (!it) { __v2MontageFinalize(); return; }
    __v2MontageRenderProgress();
    var vid = document.getElementById('cv-montage-video');
    var canvas = document.getElementById('cv-montage-canvas');
    var ctx = canvas.getContext('2d');
    var meta = document.getElementById('cv-montage-meta');
    if (meta) {
        meta.innerHTML =
            '<span class="cv-montage-time">' + __v2FmtTime(it.t_sec) + '</span>' +
            '<span class="cv-montage-kinds">' + (it.kinds || []).join(' · ') + '</span>' +
            '<span class="cv-montage-conf">match ' + Math.round((it.confidence || 0) * 100) + '%</span>';
    }
    __v2MontageFetchTrack(it);                     // ring-follow track for this touch
    if (m.queue[1]) __v2MontageFetchTrack(m.queue[1]);  // prefetch next
    var start = it.clip_start_sec, end = it.clip_end_sec;
    m.playing = true;
    function seekStart() {
        try { vid.currentTime = start; } catch (e) {}
        var p = vid.play(); if (p && p.catch) p.catch(function () {});
    }
    if (vid.readyState >= 1) seekStart();
    else vid.addEventListener('loadeddata', function ol() { vid.removeEventListener('loadeddata', ol); seekStart(); });
    vid.ontimeupdate = function () {
        if (vid.currentTime >= end || vid.currentTime < start - 0.2) {
            try { vid.currentTime = start; } catch (e) {}
        }
    };

    function draw() {
        if (!m.playing || __v2Montage !== m) return;
        var nW = vid.videoWidth, nH = vid.videoHeight;
        if (nW && nH) {
            // Show the WHOLE frame at its true aspect ratio (no square squeeze):
            // size the canvas buffer to the video's aspect the first time we know it.
            var bufW = Math.min(nW, 640);
            var bufH = Math.round(bufW * nH / nW);
            if (canvas.width !== bufW || canvas.height !== bufH) {
                canvas.width = bufW; canvas.height = bufH;
            }
            ctx.drawImage(vid, 0, 0, nW, nH, 0, 0, canvas.width, canvas.height);

            // Player position (normalized, whole-frame) + box height, so the arrow
            // sits directly above the detection box — over the player's head.
            var pos = (it.__track && it.__track.length)
                ? __v2MontagePosAt(it.__track, vid.currentTime) : null;
            var nx, boxTopN;
            var nhN = (it.__track && it.__track.length && it.__track[0].nh)
                ? it.__track[0].nh : 0.14;
            if (pos) {
                nx = pos.nx;
                boxTopN = pos.ny - nhN / 2;                  // top of the player box
            } else {
                // No tracklet → fall back to the stored contact box (640-space).
                var sc = nW / Math.min(nW, 640);
                nx = ((it.crop[0] + it.crop[2]) / 2) * sc / nW;
                boxTopN = (it.crop[1] * sc) / nH;
            }
            var ax = nx * canvas.width;
            var ay = boxTopN * canvas.height - 4;            // just above the box top
            var mw = 7, mh = 10;                             // small chevron ▼
            ctx.fillStyle = 'rgba(48,255,143,0.95)';
            ctx.strokeStyle = 'rgba(0,0,0,0.55)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(ax, ay);                              // tip points down at the head
            ctx.lineTo(ax - mw, ay - mh);
            ctx.lineTo(ax + mw, ay - mh);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }
        requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
}

// Does deciding `it` as `dec` also settle queued touch `q`?
//  • colour: "Not me" on the OTHER team clears every other same-kit opponent.
//  • appearance: within your kit, a look-alike group settles both ways (soft).
function __v2MontageMatches(it, q, dec) {
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
        cvV2Mode = true;
        if (j.state === 'review') { enterV2Review(j); return; }
        finishCvAnalysis(v2HotspotsToSegments(j.hotspots || []), 'v2');
        return;
    }
    finishCvAnalysis((j && j.segments) || [], j && j.note);
}

function finishCvAnalysis(segments, note) {
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
        video_available: true
    };
    pushMatchCatalogueEntry(catEntry);
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
        document.getElementById('res-off-markov').innerText = currentHybridResults.offMarkov;
        document.getElementById('res-off-ridge').innerText = currentHybridResults.offRidge;
        document.getElementById('res-def-markov').innerText = currentHybridResults.defMarkov;
        document.getElementById('res-def-ridge').innerText = currentHybridResults.defRidge;
        const oaVal = parseFloat(currentHybridResults.netScore);
        document.getElementById('res-overall').style.color = oaVal >= 0 ? '#4caf50' : '#f44336';
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
                label: 'Hybrid Value', data: data,
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
document.addEventListener('DOMContentLoaded', () => {
    const savedKey = localStorage.getItem('futfidget_groq_key');
    if (savedKey) toggleKeyUI(true);
});

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

function saveApiKey() {
    const input = document.getElementById('api-key-input');
    const key = input.value.trim();
    if (key.startsWith('gsk_')) {
        localStorage.setItem('futfidget_groq_key', key);
        toggleKeyUI(true);
        alert("Key saved securely to your browser!");
    } else {
        alert("Invalid Key. Groq API keys usually start with 'gsk_'");
    }
}

function clearApiKey() {
    localStorage.removeItem('futfidget_groq_key');
    document.getElementById('api-key-input').value = '';
    toggleKeyUI(false);
}

async function generateScoutReport() {
    let apiKey = localStorage.getItem('futfidget_groq_key');
    if (!apiKey) apiKey = document.getElementById('api-key-input').value;
    const outputBox = document.getElementById('ai-output');
    const btn = document.getElementById('gen-report-btn');
    if (!apiKey) { alert("Please paste or save your Groq API Key first."); return; }
    if (matchStats.length === 0 || !currentHybridResults) { outputBox.innerText = "No stats collected yet. Play a match first!"; return; }

    btn.disabled = true;
    btn.innerText = "SCOUTING...";
    outputBox.innerHTML = "<span style='color:#f2c94c'>Analyzing gameplay patterns & calculating hybrid matrices...</span>";

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
    The player position is ${position}. Their final score was ${netScore}. Their direct contributions were ${totalMarkov} (Markov score), however the long term impacts of their actions and their risks could actually be ${totalRidge} (Ridge score).

    The score is a hybrid valuation made from Markov matrices evaluating the
    immediate impact their actions have on improving xG as well as Ridge Regression
    which evaluates the long term impacts their actions have on winning and
    losing. For example, Progression may have a direct positive impact on the
    Markov level, but it has long term risks embodied by the Ridge regression
    since dribbling a lot can lead to more dispossessions.

    Here is the action count: ${JSON.stringify(statSummary)}.

    To help you analyze risk vs reward, here are the underlying coefficients used for a ${position}:
    - Markov Valuations (Immediate xG impact): ${markovValuations}
    - Ridge Valuations (Long term Win/Loss impact): ${ridgeValuations}

    Here is the chronological timeline of their actions: ${timelineString}.

    Provide a brief, bulleted report:
    1. Tactical Role (Based on actions)
    2. Key Strengths (High counts)
    3. Areas to Improve (Missing actions for this position)
    4. Possible risks of their play style (Is so much of this action actually necessary?)
    5. Possible Drills to use to Improve upon Weaknesses
    6. Mentality Changes that the Player can have in order to improve
    7. Evaluate the temporal aspect of things (did the player perform well in the first half but not the second half, did they spend a lot of time on the bench? Why might they have been subbed off?)
    8. Other Key Insights
    9. A 1-sentence summary rating.

    Keep it under 500 words. Use a professional, critical tone. (IMPORTANT DIRECTIVE: Only use this struct 9 point format for the first initial report, any subsequent questions should be directly addressed to their specific question)
    Additionally, if the player asks anything that isn't related to football and their improvement, please respond that you can't answer because you are an AI strictly used to coach football.`;

    try {
        aiChatHistory = [
            { role: "system", content: "You are a highly analytical football data scientist and scout." },
            { role: "user", content: systemPrompt }
        ];
        const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiKey}` },
            body: JSON.stringify({ model: "llama-3.3-70b-versatile", messages: aiChatHistory, temperature: 0.6 })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error.message);
        const report = data.choices[0].message.content;
        outputBox.innerText = report;
        aiChatHistory.push({ role: "assistant", content: report });
        document.getElementById('followup-container').classList.remove('hidden');
        btn.innerText = "REPORT GENERATED";
        btn.style.background = "#2e7d32";
        setTimeout(() => { btn.disabled = false; btn.innerText = "GENERATE AGAIN"; btn.style.background = "linear-gradient(135deg, #7000ff 0%, #3d0096 100%)"; }, 3000);
    } catch (error) {
        console.error(error);
        outputBox.innerText = "Error: " + error.message;
        btn.innerText = "TRY AGAIN";
        btn.disabled = false;
        if (error.message.includes("401") || error.message.includes("key")) { clearApiKey(); alert("Your API Key seems invalid. Please check it."); }
    }
}

// --- 14. HELP MODAL ---
function openHelp() { document.getElementById('help-modal').classList.remove('hidden'); }
function closeHelp() { document.getElementById('help-modal').classList.add('hidden'); }
function switchTab(tabId, btnElement) {
    document.querySelectorAll('#help-modal .tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('#help-modal .tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btnElement.classList.add('active');
}
function openGroqHelp() {
    openHelp();
    const aiTabBtn = document.querySelector("button[onclick=\"switchTab('tab-ai', this)\"]");
    if (aiTabBtn) aiTabBtn.click();
}

function escapeHtml(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// --- 15. FOLLOW-UP Q&A ---
async function askFollowUp() {
    const inputField = document.getElementById('followup-input');
    const question = inputField.value.trim();
    const outputBox = document.getElementById('ai-output');
    const askBtn = document.getElementById('followup-btn');
    let apiKey = localStorage.getItem('futfidget_groq_key');
    if (!apiKey) apiKey = document.getElementById('api-key-input').value;
    if (!question || !apiKey) return;

    aiChatHistory.push({ role: "user", content: question });
    outputBox.innerHTML += `\n\n<hr style="border-color: #333;">\n<strong style="color: #f2c94c;">YOU:</strong> ${escapeHtml(question)}\n<strong style="color: #30ff8f;">COACH AI:</strong> <em id="ai-thinking">Thinking...</em>`;
    outputBox.scrollTop = outputBox.scrollHeight;
    inputField.value = "";
    askBtn.disabled = true;
    askBtn.style.opacity = "0.5";

    try {
        const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: "llama-3.3-70b-versatile", messages: aiChatHistory, temperature: 0.6 })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error.message);
        const answer = data.choices[0].message.content;
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
