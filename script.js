/* script.js - FIXED OVERLAY, INTERACTION & AI AGENT */

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

// --- 1. CORE CHECK LOGIC ---
function checkStartReady() {
    const fileInput = document.getElementById('video-input');
    const startBtn = document.getElementById('start-btn');
    
    if (!fileInput || !startBtn) return;

    const hasVideo = fileInput.files.length > 0;
    const hasPosition = selectedPosition !== null;

    if (hasVideo && hasPosition) {
        startBtn.disabled = false;
        startBtn.style.opacity = "1";
        startBtn.style.cursor = "pointer";
        startBtn.innerText = "START ANALYSIS";
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
    document.getElementById('selected-pos-display').innerText = pos + " SELECTED";
    checkStartReady();
}

// --- 3. INITIALIZE APP ---
function initApp() {
    const fileInput = document.getElementById('video-input');
    if (!fileInput.files || !fileInput.files[0]) return;

    const oppName = document.getElementById('opponent-name').value || "Opponent";
    const scoreUs = document.getElementById('score-us') ? (parseInt(document.getElementById('score-us').value) || 0) : 0;
    const scoreThem = document.getElementById('score-them') ? (parseInt(document.getElementById('score-them').value) || 0) : 0;
    currentScore = { us: scoreUs, them: scoreThem };
    
    document.getElementById('display-match-name').innerText = "VS " + oppName.toUpperCase();
    document.getElementById('display-score').innerText = `${currentScore.us} - ${currentScore.them}`;

    // Remove Setup Screen (Prevents Blocking)
    const setupScreen = document.getElementById('setup-screen');
    if (setupScreen) {
        setupScreen.classList.add('hidden');
        setTimeout(() => setupScreen.remove(), 500); 
    }
    
    const app = document.getElementById('app-layout');
    app.classList.remove('hidden');
    app.style.display = 'flex';

    // Video Setup
    const videoPlayer = document.getElementById('main-player');
    const placeholder = document.getElementById('vid-placeholder');
    const wrapper = document.getElementById('video-wrapper');
    const slider = document.getElementById('seek-slider');

    placeholder.style.display = 'none'; 

    const fileURL = URL.createObjectURL(fileInput.files[0]);
    videoPlayer.src = fileURL;
    videoPlayer.play().catch(e => console.log("Autoplay blocked"));

    videoPlayer.addEventListener('timeupdate', updateVideoTimer);
    videoPlayer.addEventListener('loadedmetadata', () => { slider.max = videoPlayer.duration; });
    
    slider.addEventListener('mousedown', () => { isSeeking = true; });
    slider.addEventListener('mouseup', () => { isSeeking = false; });
    slider.addEventListener('input', () => { videoPlayer.currentTime = slider.value; });
    
    wrapper.addEventListener('wheel', handleWheel, { passive: false });
    wrapper.addEventListener('mousedown', startPan);
    window.addEventListener('mousemove', pan);
    window.addEventListener('mouseup', endPan);
    document.addEventListener('keydown', handleKeyShortcuts);
    
    updateSpeedConfig();
}

// --- 4. VIDEO TIMER ---
function updateVideoTimer() { 
    if (isSeeking) return;
    const videoPlayer = document.getElementById('main-player'); 
    const slider = document.getElementById('seek-slider'); 
    const timeDisplay = document.getElementById('time-display'); 
    const placeholder = document.getElementById('vid-placeholder');

    if (videoPlayer.currentTime > 0 && placeholder.style.display !== 'none') {
        placeholder.style.display = 'none';
        slider.max = videoPlayer.duration;
    }

    slider.value = videoPlayer.currentTime; 
    const m = Math.floor(videoPlayer.currentTime / 60).toString().padStart(2, '0'); 
    const s = Math.floor(videoPlayer.currentTime % 60).toString().padStart(2, '0'); 
    if(timeDisplay) timeDisplay.innerText = `${m}:${s}`; 
}

// --- 5. LOG STATS ---
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

    // 1. Save the action to memory
    const m = Math.floor(currentTime / 60).toString().padStart(2, '0'); 
    const s = Math.floor(currentTime % 60).toString().padStart(2, '0'); 
    matchStats.push({ action: actionName, timeStr: `${m}:${s}`, seconds: currentTime }); 
    
    // 2. NEW: Instantly calculate and update the Live Dashboard!
    if (typeof calculatePerformance === "function") {
        const liveResults = calculatePerformance(
            matchStats, 
            currentScore, 
            videoPlayer.duration || 90, 
            [], // Ignored for live dashboard
            selectedPosition || 'FW'
        );
        
        const elNet = document.getElementById('dash-net');
        if (elNet) {
            elNet.innerText = liveResults.netScore;
            // Turn green if positive, red if negative
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

    // 3. NEW: Automatically jump back to the Main Menu
    goBack();
    
    // 4. Visual green flash to confirm it worked
    const app = document.getElementById('app-layout');
    app.style.boxShadow = "inset 0 0 20px #30ff8f";
    setTimeout(() => { app.style.boxShadow = "none"; }, 150);

    if (navigator.vibrate) navigator.vibrate(50); 
}

// --- 6. BENCH LOGIC ---
function addBenchBlock() {
    const track = document.getElementById('bench-track');
    const id = Date.now();
    const newBlock = { id: id, startPct: 0.4, endPct: 0.5, element: null };
    const blockEl = document.createElement('div'); blockEl.className = 'bench-block-container'; blockEl.id = 'block-' + id;
    
    const fill = document.createElement('div'); fill.className = 'bench-fill';
    const leftH = document.createElement('div'); leftH.className = 'bench-handle left';
    const rightH = document.createElement('div'); rightH.className = 'bench-handle right';
    const closeBtn = document.createElement('div'); closeBtn.className = 'bench-remove'; 
    closeBtn.innerText = '×'; 
    closeBtn.onclick = () => removeBenchBlock(id);
    
    blockEl.appendChild(fill); blockEl.appendChild(leftH); blockEl.appendChild(rightH); blockEl.appendChild(closeBtn);
    track.appendChild(blockEl);
    
    newBlock.element = blockEl; 
    benchBlocks.push(newBlock);
    
    setupBlockListeners(newBlock, leftH, rightH); 
    renderBlock(newBlock);
}

function removeBenchBlock(id) { 
    const index = benchBlocks.findIndex(b => b.id === id); 
    if (index > -1) { 
        benchBlocks[index].element.remove(); 
        benchBlocks.splice(index, 1); 
    } 
}

function setupBlockListeners(blockObj, leftBtn, rightBtn) {
    const track = document.getElementById('bench-track');
    
    // 1. Logic for stretching the left and right edges
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
        }; 
        window.addEventListener('mousemove', move); 
        window.addEventListener('mouseup', stop); 
    };
    
    leftBtn.addEventListener('mousedown', (e) => startEdgeDrag(e, true)); 
    rightBtn.addEventListener('mousedown', (e) => startEdgeDrag(e, false));

    // 2. NEW: Logic for clicking and dragging the entire block
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

            // Prevent the block from being dragged off the edges of the track
            if (newStart < 0) {
                newStart = 0;
                newEnd = blockWidthPct;
            } else if (newEnd > 1) {
                newEnd = 1;
                newStart = 1 - blockWidthPct;
            }

            blockObj.startPct = newStart;
            blockObj.endPct = newEnd;
            renderBlock(blockObj);
        };

        const stopBlock = () => {
            window.removeEventListener('mousemove', moveBlock);
            window.removeEventListener('mouseup', stopBlock);
        };

        window.addEventListener('mousemove', moveBlock);
        window.addEventListener('mouseup', stopBlock);
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

// --- 7. VIDEO CONTROLS ---
function updateSpeedConfig() { 
    const slowInput = document.getElementById('slow-speed-set');
    const fastInput = document.getElementById('fast-speed-set');
    if(slowInput) slowSpeed = parseFloat(slowInput.value); 
    if(fastInput) fastSpeed = parseFloat(fastInput.value); 
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
    if(display) display.innerText = rate + 'x'; 
}

function handleKeyShortcuts(e) { 
    // NEW: If the user is typing in an input field (like the chat bar), ignore shortcuts!
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const videoPlayer = document.getElementById('main-player'); 
    if (!videoPlayer) return; 
    switch(e.code) { 
        case 'Space': e.preventDefault(); togglePlay(); break; 
        case 'ArrowLeft': videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 5); break; 
        case 'ArrowRight': videoPlayer.currentTime = Math.min(videoPlayer.duration, videoPlayer.currentTime + 5); break; 
        case 'KeyS': toggleCustomSpeed(); break; 
    } 
}

function togglePlay() { 
    const video = document.getElementById('main-player'); 
    const btn = document.getElementById('play-pause-btn'); 
    if (video.paused) { video.play(); btn.innerText = "⏸"; } 
    else { video.pause(); btn.innerText = "▶"; } 
}

// --- 8. ZOOM & PAN ---
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
function resetZoom() { 
    zoomLevel = 1; panX = 0; panY = 0; applyTransform(); 
}

// --- 9. MENUS (FIXED TOGGLING) ---
function openSubMenu(type) { 
    const mainMenu = document.getElementById('menu-main');
    mainMenu.style.display = 'none';
    mainMenu.classList.add('hidden'); 

    const targetMenu = document.getElementById('menu-' + type);
    targetMenu.classList.remove('hidden'); 
    targetMenu.style.display = 'grid'; 
}

function goBack() { 
    document.querySelectorAll('.sub-menu').forEach(el => {
        el.style.display = 'none';
        el.classList.add('hidden');
    }); 

    const mainMenu = document.getElementById('menu-main');
    mainMenu.classList.remove('hidden');
    mainMenu.style.display = 'grid'; 
}

// --- 10. FINISH MATCH (UPDATED FOR HYBRID DATA) ---
let currentHybridResults = null; // Store globally so the AI can access it

function finishMatch() {
    const videoPlayer = document.getElementById('main-player');
    videoPlayer.pause();
    const duration = videoPlayer.duration || 0;
    
    let excludedRanges = benchBlocks.map(b => ({ start: b.startPct * duration, end: b.endPct * duration }));

    document.getElementById('app-layout').classList.add('hidden');
    document.getElementById('app-layout').style.display = 'none';
    
    const resScreen = document.getElementById('results-screen');
    resScreen.classList.remove('hidden');
    resScreen.style.display = 'flex';

    if (typeof calculatePerformance === "function") {
        currentHybridResults = calculatePerformance(matchStats, currentScore, duration, excludedRanges, selectedPosition);

        document.getElementById('result-header').innerText = `PERFORMANCE REPORT (${selectedPosition})`;
        
        // --- NEW: Update Goals and Assists Count ---
        const totalGoals = matchStats.filter(s => s.action === 'Goal').length;
        const totalAssists = matchStats.filter(s => s.action === 'Assist').length;
        document.getElementById('res-goals').innerText = totalGoals;
        document.getElementById('res-assists').innerText = totalAssists;

        // Update all 5 Dashboard Numbers
        document.getElementById('res-overall').innerText = currentHybridResults.netScore;
        document.getElementById('res-off-markov').innerText = currentHybridResults.offMarkov;
        document.getElementById('res-off-ridge').innerText = currentHybridResults.offRidge;
        document.getElementById('res-def-markov').innerText = currentHybridResults.defMarkov;
        document.getElementById('res-def-ridge').innerText = currentHybridResults.defRidge;
        
        const oaVal = parseFloat(currentHybridResults.netScore);
        document.getElementById('res-overall').style.color = oaVal >= 0 ? '#4caf50' : '#f44336';

        renderWPAChart(currentHybridResults.chartData, duration, excludedRanges);
    } else {
        alert("Error: calculations.js is not loaded correctly.");
    }
}

// --- 11. CHARTING ---
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
                label: 'Hybrid Value',
                data: data,
                borderColor: '#f2c94c', 
                backgroundColor: 'rgba(242, 201, 76, 0.1)',
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

// --- 12. AI AGENT INTEGRATION (GROQ + HYBRID PROMPT) ---

// UI functionality for Saving/Loading Key
document.addEventListener('DOMContentLoaded', () => {
    const savedKey = localStorage.getItem('futfidget_groq_key');
    if (savedKey) toggleKeyUI(true);
});

function toggleKeyUI(isSaved) {
    const inputContainer = document.getElementById('api-key-container');
    const savedBadge = document.getElementById('api-key-saved');
    if (isSaved) {
        if(inputContainer) { inputContainer.classList.add('hidden'); inputContainer.style.display = 'none'; }
        if(savedBadge) { savedBadge.classList.remove('hidden'); savedBadge.style.display = 'flex'; }
    } else {
        if(inputContainer) { inputContainer.classList.remove('hidden'); inputContainer.style.display = 'flex'; }
        if(savedBadge) { savedBadge.classList.add('hidden'); savedBadge.style.display = 'none'; }
    }
}

function saveApiKey() {
    const input = document.getElementById('api-key-input');
    const key = input.value.trim();
    if (key.startsWith('gsk_')) { // Groq keys start with gsk_
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

// Main AI Generation
async function generateScoutReport() {
    let apiKey = localStorage.getItem('futfidget_groq_key'); 
    if (!apiKey) apiKey = document.getElementById('api-key-input').value;

    const outputBox = document.getElementById('ai-output');
    const btn = document.getElementById('gen-report-btn');

    if (!apiKey) {
        alert("Please paste or save your Groq API Key first.");
        return;
    }
    if (matchStats.length === 0 || !currentHybridResults) {
        outputBox.innerText = "No stats collected yet. Play a match first!";
        return;
    }

    btn.disabled = true;
    btn.innerText = "SCOUTING...";
    outputBox.innerHTML = "<span style='color:#f2c94c'>Analyzing gameplay patterns & calculating hybrid matrices...</span>";

    // Prepare Summary and Timeline
    const statSummary = matchStats.reduce((acc, curr) => {
        acc[curr.action] = (acc[curr.action] || 0) + 1;
        return acc;
    }, {});
    const timelineString = matchStats.map(s => `[${s.timeStr}] ${s.action}`).join(', ');

    const position = selectedPosition || "Player";
    
    // Sum up totals for the prompt
    const totalMarkov = (parseFloat(currentHybridResults.offMarkov) + parseFloat(currentHybridResults.defMarkov)).toFixed(2);
    const totalRidge = (parseFloat(currentHybridResults.offRidge) + parseFloat(currentHybridResults.defRidge)).toFixed(2);
    const netScore = currentHybridResults.netScore;

    // Get the base valuations used for this specific position
    const markovValuations = JSON.stringify(currentHybridResults.coeffMarkov);
    const ridgeValuations = JSON.stringify(currentHybridResults.coeffRidge);

    // EXACT CUSTOM PROMPT
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
        // 1. Save the initial prompt to memory
        aiChatHistory = [
            { role: "system", content: "You are a highly analytical football data scientist and scout." },
            { role: "user", content: systemPrompt }
        ];

        const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model: "llama-3.3-70b-versatile", 
                messages: aiChatHistory, // <-- Uses the memory array
                temperature: 0.6
            })
        });

        const data = await response.json();

        if (data.error) throw new Error(data.error.message);

        const report = data.choices[0].message.content;
        outputBox.innerText = report;
        
        // 2. Save AI's response to memory
        aiChatHistory.push({ role: "assistant", content: report });
        
        // 3. Unhide the follow-up chat bar
        document.getElementById('followup-container').classList.remove('hidden');

        btn.innerText = "REPORT GENERATED";
        btn.style.background = "#2e7d32"; 
        
        setTimeout(() => {
            btn.disabled = false;
            btn.innerText = "GENERATE AGAIN";
            btn.style.background = "linear-gradient(135deg, #7000ff 0%, #3d0096 100%)";
        }, 3000);

    } catch (error) {
        console.error(error);
        outputBox.innerText = "Error: " + error.message;
        btn.innerText = "TRY AGAIN";
        btn.disabled = false;
        
        if (error.message.includes("401") || error.message.includes("key")) {
            clearApiKey();
            alert("Your API Key seems invalid. Please check it.");
        }
    }
}
// --- HELP MODAL LOGIC ---
function openHelp() {
    document.getElementById('help-modal').classList.remove('hidden');
}

function closeHelp() {
    document.getElementById('help-modal').classList.add('hidden');
}

function switchTab(tabId, btnElement) {
    // Hide all tabs
    document.querySelectorAll('#help-modal .tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('#help-modal .tab-btn').forEach(b => b.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(tabId).classList.add('active');
    btnElement.classList.add('active');
}

function openGroqHelp() {
    // 1. Open the modal normally
    openHelp(); 
    
    // 2. Find the "Connecting AI" tab button and automatically click it
    const aiTabBtn = document.querySelector("button[onclick=\"switchTab('tab-ai', this)\"]");
    if (aiTabBtn) {
        aiTabBtn.click();
    }
}

// --- FOLLOW-UP Q&A LOGIC ---
async function askFollowUp() {
    const inputField = document.getElementById('followup-input');
    const question = inputField.value.trim();
    const outputBox = document.getElementById('ai-output');
    const askBtn = document.getElementById('followup-btn');
    
    let apiKey = localStorage.getItem('futfidget_groq_key'); 
    if (!apiKey) apiKey = document.getElementById('api-key-input').value;

    if (!question || !apiKey) return;

    // 1. Save user question to memory & update UI
    aiChatHistory.push({ role: "user", content: question });
    
    // Add horizontal line and the new message to the chat box
    outputBox.innerHTML += `\n\n<hr style="border-color: #333;">\n<strong style="color: #f2c94c;">YOU:</strong> ${question}\n<strong style="color: #30ff8f;">COACH AI:</strong> <em id="ai-thinking">Thinking...</em>`;
    
    // Auto-scroll to bottom
    outputBox.scrollTop = outputBox.scrollHeight;
    
    // Disable input while waiting
    inputField.value = "";
    askBtn.disabled = true;
    askBtn.style.opacity = "0.5";

    try {
        // 2. Send the ENTIRE memory array back to Groq
        const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: "llama-3.3-70b-versatile", // Matches your main function
                messages: aiChatHistory,
                temperature: 0.6
            })
        });

        const data = await response.json();
        if (data.error) throw new Error(data.error.message);

        const answer = data.choices[0].message.content;
        
        // 3. Save new answer to memory
        aiChatHistory.push({ role: "assistant", content: answer });
        
        // 4. Replace "Thinking..." with the actual answer safely
        const thinkingEl = document.getElementById('ai-thinking');
        if (thinkingEl) {
            thinkingEl.outerHTML = answer; // Replaces the <em> tag with the actual text
        }

    } catch (error) {
        console.error(error);
        const thinkingEl = document.getElementById('ai-thinking');
        if (thinkingEl) {
            thinkingEl.outerHTML = `<span style="color: #ff2e4d;">Error: ${error.message}</span>`;
        }
    } finally {
        // Re-enable input
        askBtn.disabled = false;
        askBtn.style.opacity = "1";
        outputBox.scrollTop = outputBox.scrollHeight;
    }
}

// Allow pressing "Enter" to send the message
document.addEventListener('DOMContentLoaded', () => {
    const fInput = document.getElementById('followup-input');
    if (fInput) {
        fInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                askFollowUp();
            }
        });
    }
});