// help.js — How to Use modal (definitions, interface, scoring, Groq API key)

const helpModalHTML = `
    <button type="button" onclick="openHelp()" class="help-btn-top" aria-label="How to use PolyFut">HOW TO USE</button>

    <div id="help-modal" class="hidden" role="dialog" aria-modal="true" aria-labelledby="help-modal-title">
        <div class="help-content">
            <button type="button" class="help-close" onclick="closeHelp()" aria-label="Close help">&times;</button>
            <h2 id="help-modal-title">How to Use PolyFut</h2>

            <div class="help-tabs">
                <button type="button" class="tab-btn active" onclick="switchTab('tab-definitions', this)">1. Definitions</button>
                <button type="button" class="tab-btn" onclick="switchTab('tab-interface', this)">2. Interface</button>
                <button type="button" class="tab-btn" onclick="switchTab('tab-calculations', this)">3. Calculations</button>
                <button type="button" class="tab-btn" onclick="switchTab('tab-ai', this)">4. AI Report</button>
                <button type="button" class="tab-btn" onclick="switchTab('tab-steps', this)">5. Next Steps</button>
            </div>

            <div id="tab-definitions" class="tab-content active">
                <h4>Threat Points (TP)</h4>
                <ul>
                    <li><strong>TP = Threat Points</strong> — PolyFut's scoring unit for how much an action helped (or hurt) your team.</li>
                    <li><strong>Scale:</strong> 1 TP ≈ 1% of a goal. A goal is worth <strong>100 TP</strong>.</li>
                    <li><strong>Net Threat:</strong> Your overall score after combining impact (chance creation / chance prevention) and risk (mistakes, volume that leads to turnovers).</li>
                </ul>

                <h4>Shooting &amp; Finishing</h4>
                <ul>
                    <li><strong>Shot Taken:</strong> Deliberate attempt to score a goal with foot or head, excluding accidental crosses.</li>
                    <li><strong>Goal:</strong> A shot that legally and completely crosses the opponent's goal line.</li>
                    <li><strong>Assist:</strong> The final pass that directly leads to a teammate scoring a goal.</li>
                </ul>

                <h4>Passing &amp; Playmaking</h4>
                <ul>
                    <li><strong>Progression (Pass):</strong> Completed forward pass moving the ball significantly closer to the opponent's goal.</li>
                    <li><strong>Key Pass:</strong> A pass that directly leads to a teammate taking a shot, regardless of outcome.</li>
                    <li><strong>Pass into Box:</strong> Completed pass originating outside the penalty area and successfully received inside it.</li>
                    <li><strong>Cross into Box:</strong> Pass played from the wide flank areas into the center of the penalty area.</li>
                </ul>

                <h4>Driving &amp; Possession</h4>
                <ul>
                    <li><strong>Progression (Carry):</strong> Running with the ball at the feet to move significantly closer to the opponent's goal.</li>
                    <li><strong>Dribble (Beat Man):</strong> Successfully using skill or pace to get past an active defender while maintaining possession.</li>
                    <li><strong>Ball Recovery:</strong> Reacting fastest to gain possession of a loose ball that neither team clearly controlled.</li>
                </ul>

                <h4>Defending &amp; Ball Winning</h4>
                <ul>
                    <li><strong>Interception:</strong> Reading the play to cut out and steal an opponent's pass while it is traveling.</li>
                    <li><strong>High Press Win:</strong> Winning the ball back via tackle or interception in the attacking third of the pitch.</li>
                    <li><strong>Midfield Tackle:</strong> Successfully dispossessing an opponent who has the ball in the middle third of the pitch.</li>
                    <li><strong>Deep Tackle:</strong> Successfully dispossessing an opponent in your own defensive third, close to your goal.</li>
                    <li><strong>Block:</strong> Physically stepping in the way of an opponent's shot to prevent it from reaching goal.</li>
                    <li><strong>Aerial Duel Won:</strong> Winning a contested header against an opponent to pass or clear the ball.</li>
                </ul>

                <h4>Mistakes &amp; Risks</h4>
                <ul>
                    <li><strong>Dispossessed:</strong> Losing control of the ball after being successfully tackled by an opposing player.</li>
                    <li><strong>Defensive Error:</strong> A catastrophic mistake, like a bad pass or slip, gifting the opponent a high-danger chance.</li>
                    <li><strong>Foul Committed:</strong> An illegal physical challenge resulting in the referee stopping play for a free kick.</li>
                </ul>
            </div>

            <div id="tab-interface" class="tab-content">
                <h4>1. Match Setup</h4>
                <ul>
                    <li><strong>Select Position:</strong> Click on the pitch map to choose the position you are playing (Forward [FW], Midfielder [MF], or Defender [DF]). The engine will adapt its scoring model based on your choice.</li>
                    <li><strong>Upload Video:</strong> Click the upload button to load your match video. You can convert YouTube URLs to mp4 using <a href="https://turboscribe.ai/downloader/youtube/mp4" target="_blank" rel="noopener">TurboScribe</a> (or any other format converter).</li>
                    <li><strong>Pick your team:</strong> After upload, choose which kit colour you played in.</li>
                    <li><strong>Playing time (if shown):</strong> Tell PolyFut when you were actually on the pitch so analysis stays inside those windows.</li>
                    <li><strong>Tap yourself:</strong> In the four short clips, tap the marker on <strong>you</strong> so the app learns who to follow.</li>
                    <li><strong>Start analysis:</strong> Once seeding is done, click <strong>Find My Touches</strong>. Review uncertain clips, then log actions in the video player.</li>
                    <li><strong>Processing time:</strong> Full matches are analysed locally on your CPU. A long game can take a while — this is normal. You can turn off the display, lock the screen, or close the browser tab; analysis keeps running until you choose Cancel &amp; discard. Only a full shutdown or system sleep will stop it. Keep your laptop plugged in.</li>
                </ul>

                <h4>2. Tracking &amp; Playback</h4>
                <ul>
                    <li><strong>Video Controls:</strong> Play/pause (Space bar), or skip forward/backward (Arrow keys) by 5 seconds to navigate the match.</li>
                    <li><strong>Speed:</strong> Press "S" to toggle between slow and fast playback speeds. Use the speed dropdowns to configure both speeds.</li>
                    <li><strong>Zoom &amp; Pan:</strong> Scroll the mouse wheel over the video to zoom in or out. Click and drag to pan around the zoomed view. Click RESET ZOOM to return to normal.</li>
                    <li><strong>Log Actions:</strong> When you perform a key action on the pitch, pause the video and click the specific category and action (e.g., "Pass" → "Cross into Box").</li>
                    <li><strong>Manage Substitutions:</strong> If you are subbed off, add a <strong>BENCH (SUB)</strong> block to signal that you were off the pitch so it doesn't dilute your results.</li>
                </ul>

                <h4>3. Monitoring Results</h4>
                <ul>
                    <li><strong>Live Dashboard:</strong> Your <strong>Net Threat (TP)</strong> score updates as you log actions.</li>
                    <li><strong>Performance Chart:</strong> Watch the live line chart plot your positive and negative momentum over the course of the match.</li>
                    <li><strong>Match Analysis page:</strong> After FINISH, you get goals/assists, Net Threat Points, offense/defense breakdown, and an optional AI scout report.</li>
                </ul>
            </div>

            <div id="tab-calculations" class="tab-content">
                <h4>The Hybrid Valuation Engine</h4>
                <p>This engine is a completely custom model built on professional sports data science principles, combining two major predictive systems:</p>
                <ul>
                    <li><strong>What do we do?</strong> PolyFut is built for high school, middle school, and college players who want to take initiative in analysing their own performance without needing professional tools. Action values are derived from Premier League 2024-2025 data processed through Machine Learning techniques.</li>
                    <li><strong>Threat Points (TP):</strong> 1 TP = 1% of a goal (a goal = 100 TP). Your Match Analysis score is reported in TP.</li>
                    <li><strong>Markov Chains (Immediate Threat):</strong> Values how much an action immediately increases the probability of scoring an Expected Goals (xG).</li>
                    <li><strong>Ridge Regression (Long-Term Win%):</strong> Punishes mistakes and values actions that help a team maintain control and win over 90 minutes.</li>
                    <li><strong>Shadow xG Multipliers:</strong> Solves the famous "defensive bias" in football data by assigning defenders the value of the offensive chances they destroy.</li>
                </ul>
            </div>

            <div id="tab-ai" class="tab-content">
                <h4>AI scout reports</h4>
                <p>Reports work straight away — there is nothing to set up. When you click <strong>GENERATE REPORT</strong>, PolyFut sends your match statistics (the numbers you logged, not your video) to its AI service and writes you a coach-level breakdown. You can ask follow-up questions in the box below the report.</p>

                <h4>What leaves your machine, and when</h4>
                <p>Almost everything PolyFut does runs locally. Three things do not, and all are worth knowing about:</p>
                <ul>
                    <li><strong>Your report request</strong> — only when you click Generate. It contains your logged match statistics. Your video is never uploaded.</li>
                    <li><strong>A few still frames, during setup</strong> — to work out the two teams' kit colours, PolyFut may send up to three small stills from your match to the same AI service. This happens once per video, automatically, because getting the kit colours right is what lets it tell your team from the opposition. If it is unavailable, PolyFut works the colours out on your own machine instead and carries on.</li>
                    <li><strong>A version check, when you open PolyFut</strong> &mdash; a single request to polyfut.com asking what the newest version is, so the app can tell you when an update exists. It sends nothing about you or your match, downloads nothing, and installs nothing &mdash; it only shows a notice you can dismiss. Turn it off with <code>POLYFUT_UPDATE_CHECK=0</code>, or <code>"update_check": false</code> in <code>ai_config.json</code>.</li>
                </ul>
                <div class="help-setting">
                    <label class="help-toggle">
                        <input type="checkbox" id="help-kit-vision" onchange="setKitVision(this.checked)">
                        <span><strong>Send stills to read kit colours</strong></span>
                    </label>
                    <p class="help-setting-note" id="help-kit-vision-note">Loading&hellip;</p>
                </div>
                <p>Turn this off to keep every frame on your machine. Reports stay available on demand &mdash; they send your logged statistics, never the video. With it off, PolyFut works the kit colours out locally, which is less reliable on small or distant players but never leaves the machine.</p>

                <h4>If the AI is busy</h4>
                <p>The AI allowance is shared between everyone using PolyFut, so it can run out. Reports will say so; kit colours quietly fall back to the local method and analysis continues as normal.</p>
                <p>To avoid the shared queue you can add your own free Groq key, which PolyFut will use whenever the shared allowance is spent. It is stored on your device and never sent anywhere except Groq.</p>
                <ul>
                    <li><strong>Step 1:</strong> Go to the <a href="https://console.groq.com/keys" target="_blank" rel="noopener">Groq API Console</a> and log in using any method you prefer.<br>
                        <img class="help-img" src="GroqSetup1.png" alt="Groq login screen">
                        <img class="help-img" src="GroqSetup2.png" alt="Groq console screen">
                    </li>
                    <li><strong>Step 2:</strong> Click <strong>"Create API Key"</strong>.<br>
                        <img class="help-img" src="GroqSetup3.png" alt="Create API Key button">
                    </li>
                    <li><strong>Step 3:</strong> Give it a name (e.g. "PolyFut1") and submit.<br>
                        <img class="help-img" src="GroqSetup4.png" alt="Naming the API key">
                    </li>
                    <li><strong>Step 4:</strong> Copy the key — it starts with <code>gsk_</code>.<br>
                        <img class="help-img" src="GroqSetup5.png" alt="Copying the API key">
                    </li>
                    <li><strong>Step 5:</strong> On the <strong>Match Analysis</strong> page, paste it into the <strong>API Key</strong> field at the top of the AI Scout Report panel and click <strong>SAVE</strong>.<br>
                        <img class="help-img" src="GroqSetup6.png" alt="Pasting the key into PolyFut">
                    </li>
                    <li style="margin-top: 10px;"><strong>First launch:</strong> PolyFut offers this on opening too. Skipping it is fine — reports work without it.</li>
                </ul>
            </div>

            <div id="tab-steps" class="tab-content">
                <h4>Next Steps</h4>
                <p>PolyFut is designed to grow with you. Longer-term work includes user accounts for progress tracking and comparative valuations against professionals and peers.</p>
                <ul>
                    <li><strong>Sending Feedback:</strong> Go to <a href="https://forms.gle/zdpUEc1exkUhDdfp7" target="_blank" rel="noopener">this Link</a> to submit relevant feedback for our website.</li>
                </ul>
            </div>
        </div>
    </div>
`;

// The kit-colour read sends stills off the machine and is on by default, so the
// app has to be able to refuse it from inside the app. It used to require
// editing ai_config.json in the install directory, which an all-users install
// makes read-only - the users least able to edit it were the ones who could not
// opt out at all.
function pfRenderKitVision(state) {
    const box = document.getElementById('help-kit-vision');
    const note = document.getElementById('help-kit-vision-note');
    if (!box || !note) return;
    if (!state) {
        box.disabled = true;
        note.textContent = 'Could not reach PolyFut to read this setting.';
        return;
    }
    box.checked = !!state.kit_vision;
    // Disabled for a stated reason, never silently: a dead control with no
    // explanation reads as a bug.
    box.disabled = !state.available || !!state.locked;
    if (!state.available) {
        note.textContent = 'No AI service is set up on this install, so no frames '
            + 'are sent either way. Kit colours are always worked out locally.';
    } else if (state.locked) {
        note.textContent = 'Pinned by the POLYFUT_KIT_VISION environment variable, '
            + 'so it cannot be changed here.';
    } else if (state.kit_vision) {
        note.textContent = 'On. Up to three small stills per video are sent once, '
            + 'during setup, to read the two kit colours.';
    } else {
        note.textContent = 'Off. Every frame stays on your machine; kit colours are '
            + 'worked out locally.';
    }
}

function pfLoadKitVision() {
    const note = document.getElementById('help-kit-vision-note');
    if (note) note.textContent = 'Checking…';
    fetch(cvApiUrl('/api/settings'))
        .then(function (r) { return r.json(); })
        .then(pfRenderKitVision)
        .catch(function () { pfRenderKitVision(null); });
}

function setKitVision(on) {
    const note = document.getElementById('help-kit-vision-note');
    if (note) note.textContent = 'Saving…';
    fetch(cvApiUrl('/api/settings'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kit_vision: !!on })
    })
        .then(function (r) { return r.json(); })
        // Render from what the SERVER reports, not from what was clicked, so a
        // refused or failed save cannot leave the checkbox showing a state the
        // app is not actually in.
        .then(pfRenderKitVision)
        .catch(function () { pfRenderKitVision(null); });
}

function openHelp() {
    const modal = document.getElementById('help-modal');
    if (!modal) return;
    modal.classList.remove('hidden');
    document.body.classList.add('help-open');
    pfLoadKitVision();
}

function closeHelp() {
    const modal = document.getElementById('help-modal');
    if (!modal) return;
    modal.classList.add('hidden');
    document.body.classList.remove('help-open');
}

function switchTab(tabId, btn) {
    document.querySelectorAll('#help-modal .tab-content').forEach(function (el) {
        el.classList.remove('active');
    });
    document.querySelectorAll('#help-modal .tab-btn').forEach(function (el) {
        el.classList.remove('active');
    });
    const panel = document.getElementById(tabId);
    if (panel) panel.classList.add('active');
    if (btn) btn.classList.add('active');
}

document.addEventListener('DOMContentLoaded', function () {
    document.body.insertAdjacentHTML('beforeend', helpModalHTML);

    // Hide optional setup screenshots that aren't shipped with this build.
    document.querySelectorAll('#help-modal .help-img').forEach(function (img) {
        img.addEventListener('error', function () {
            img.style.display = 'none';
        });
    });

    const modal = document.getElementById('help-modal');
    if (modal) {
        modal.addEventListener('click', function (e) {
            if (e.target === modal) closeHelp();
        });
    }
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') closeHelp();
    });
});
