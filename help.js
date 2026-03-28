// help.js - Stores the Help Modal content to keep index.html clean

const helpModalHTML = `
    <button onclick="openHelp()" class="help-btn-top">HOW TO USE</button>
    
    <div id="help-modal" class="hidden">
        <div class="help-content">
            <span class="help-close" onclick="closeHelp()">&times;</span>
            <h2 style="margin-top: 0; color: #fff; border-bottom: 1px solid #333; padding-bottom: 10px;">How to Use PolyFut</h2>
            
            <div class="help-tabs">
                <button class="tab-btn active" onclick="switchTab('tab-definitions', this)">1. Definitions</button>
                <button class="tab-btn" onclick="switchTab('tab-interface', this)">2. Interface</button>
                <button class="tab-btn" onclick="switchTab('tab-calculations', this)">3. Calculations</button>
                <button class="tab-btn" onclick="switchTab('tab-ai', this)">4. Connecting AI</button>
                <button class="tab-btn" onclick="switchTab('tab-steps', this)">5. Next Steps</button>
            </div>

            <div id="tab-definitions" class="tab-content active">
                <h4>Shooting & Finishing</h4>
                <ul>
                    <li><strong>Shot Taken:</strong> Deliberate attempt to score a goal with foot or head, excluding accidental crosses.</li>
                    <li><strong>Goal:</strong> A shot that legally and completely crosses the opponent's goal line.</li>
                    <li><strong>Assist:</strong> The final pass that directly leads to a teammate scoring a goal.</li>
                </ul>

                <h4>Passing & Playmaking</h4>
                <ul>
                    <li><strong>Progression (Pass):</strong> Completed forward pass moving the ball significantly closer to the opponent's goal.</li>
                    <li><strong>Key Pass:</strong> A pass that directly leads to a teammate taking a shot, regardless of outcome.</li>
                    <li><strong>Pass into Box:</strong> Completed pass originating outside the penalty area and successfully received inside it.</li>
                    <li><strong>Cross into Box:</strong> Pass played from the wide flank areas into the center of the penalty area.</li>
                </ul>

                <h4>Driving & Possession</h4>
                <ul>
                    <li><strong>Progression (Carry):</strong> Running with the ball at the feet to move significantly closer to the opponent's goal.</li>
                    <li><strong>Dribble (Beat Man):</strong> Successfully using skill or pace to get past an active defender while maintaining possession.</li>
                    <li><strong>Ball Recovery:</strong> Reacting fastest to gain possession of a loose ball that neither team clearly controlled.</li>
                </ul>

                <h4>Defending & Ball Winning</h4>
                <ul>
                    <li><strong>Interception:</strong> Reading the play to cut out and steal an opponent's pass while it is traveling.</li>
                    <li><strong>High Press Win:</strong> Winning the ball back via tackle or interception in the attacking third of the pitch.</li>
                    <li><strong>Midfield Tackle:</strong> Successfully dispossessing an opponent who has the ball in the middle third of the pitch.</li>
                    <li><strong>Deep Tackle:</strong> Successfully dispossessing an opponent in your own defensive third, close to your goal.</li>
                    <li><strong>Block:</strong> Physically stepping in the way of an opponent's shot to prevent it from reaching goal.</li>
                    <li><strong>Aerial Duel Won:</strong> Winning a contested header against an opponent to pass or clear the ball.</li>
                </ul>

                <h4>Mistakes & Risks</h4>
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
                    <li><strong>Upload Video:</strong> Click the upload button to load your match video (.mp4 format). You can convert YouTube URLs to mp4 using <a href="https://turboscribe.ai/downloader/youtube/mp4" target="_blank" style="color: #f2c94c;">TurboScribe</a> (or any other format converter).</li>
                    <li><strong>Start Analysis:</strong> Once your position is selected and the video is loaded, click the "START ANALYSIS" button to enter the main dashboard.</li>
                </ul>
                
                <h4>2. Tracking & Playback</h4>
                <ul>
                    <li><strong>Video Controls:</strong> Play/pause (Space bar), or skip forward/backward (Arrow keys) by 10 seconds to navigate the match. Click "S" to toggle the playback speed; you can adjust how fast or how slow the video goes.</li>
                    <li><strong>Log Actions:</strong> When you perform a key action on the pitch, pause the video. Click the specific category and action you performed (e.g., "Pass" -> "Cross into Box").</li>
                    <li><strong>Manage Substitutions:</strong> If you are subbed off, add a <strong>BENCH (SUB)</strong> block to signal to the AI that you were off the pitch so it doesn't dilute your results.</li>
                </ul>

                <h4>3. Monitoring Results</h4>
                <ul>
                    <li><strong>Live Dashboard:</strong> Your <strong>Net Impact</strong> score (calculating your xG added and Risk factor) will update instantly with every action you log.</li>
                    <li><strong>Performance Chart:</strong> Watch the live line chart plot your positive and negative momentum over the course of the match.</li>
                </ul>
            </div>

            <div id="tab-calculations" class="tab-content">
                <h4>The Hybrid Valuation Engine</h4>
                <p>This engine is a completely custom model built on professional sports data science principles, combining two major predictive systems:</p>
                <ul>
                    <li><strong>What do we do?</strong> I am a highschool Varsity Football player that believes that modern sporting data science had been disproportionately catered towards high level engineers, analysts and professionals rather than students and academy players. 
                    This project is meant to act as a basis to allow for Middleschoolers, Highschoolers and College Students to take initiative regarding their own improvement in Football, through a simple and straightforward interface.
                    Through our methods, players are clearly able to see the contributions they make on the field and have an AI system specialised towards helping them improve. The valuation of actions are completed by processing
                    Premier League Stats from the 2024-2025 season through Machine Learning techniques. Some of the related techniques we used are mentioned below.
                    </li>
                    <li><strong>Markov Chains (Immediate Threat):</strong> Values how much an action immediately increases the probability of scoring an Expected Goals (xG).</li>
                    <li><strong>Ridge Regression (Long-Term Win%):</strong> Punishes mistakes and values actions that help a team maintain control and win over 90 minutes.</li>
                    <li><strong>Shadow xG Multipliers:</strong> Solves the famous "defensive bias" in football data by assigning defenders the value of the offensive chances they destroy.</li>
                    
                </ul>
            </div>

            <div id="tab-ai" class="tab-content">
                <h4>Connecting the AI</h4>
                <p style="margin-bottom: 5px; color: #a0a0b0;">To get your AI scouting report, you need a free Groq API key:</p>
                <ul>
                    <li><strong>Step 1:</strong> Go to the <a href="https://console.groq.com/keys" target="_blank" style="color: #f2c94c;">Groq API Console</a> and log in using any method you prefer.<br>
                        <img src="GroqSetup1.png" alt="Groq Login Screen" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; border: 1px solid #333; box-shadow: 0 4px 10px rgba(0,0,0,0.5);">
                        <img src="GroqSetup2.png" alt="Groq Console Screen" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; border: 1px solid #333; box-shadow: 0 4px 10px rgba(0,0,0,0.5);">
                    </li>
                    
                    <li><strong>Step 2:</strong> Click the <strong>"Create API Key"</strong> button.<br>
                        <img src="GroqSetup3.png" alt="Create API Key Button" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; border: 1px solid #333; box-shadow: 0 4px 10px rgba(0,0,0,0.5);">
                    </li>
                    
                    <li><strong>Step 3:</strong> Give your API key a name (e.g., "PolyFut1") and click submit.<br>
                        <img src="GroqSetup4.png" alt="Name API Key" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; border: 1px solid #333; box-shadow: 0 4px 10px rgba(0,0,0,0.5);">
                    </li>

                    <li><strong>Step 4:</strong> Copy the generated API key.<br>
                        <img src="GroqSetup5.png" alt="Copy API Key" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; border: 1px solid #333; box-shadow: 0 4px 10px rgba(0,0,0,0.5);">
                    </li>
                    
                    <li><strong>Step 5:</strong> Paste it into the <strong>API Key Slot</strong> at the bottom of the PolyFut dashboard, and save it.<br>
                        <img src="GroqSetup6.png" alt="Paste in PolyFut" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; border: 1px solid #333; box-shadow: 0 4px 10px rgba(0,0,0,0.5);">
                    </li>
                    
                    <li style="margin-top: 10px;"><strong>Get Feedback:</strong> Once your key is saved, click <strong>GENERATE REPORT</strong> to have the AI analyze your match data and write a professional, coach-level breakdown!</li>
                </ul>
            </div>

            <div id="tab-steps" class="tab-content">
                <h4>Next Steps</h4>
                <p> I am currently looking into furthering the capacity of PolyFut by figuring out a way to implement computer-vision capabilities into the website to allow for a smoother analysis process.</p>
                <p> Furthermore, I'm trying to find a way for users to be able to setup accounts to track their long term progress.</p>
                <p> Lastly, I'm working on comparative valuations so that users can see their performances relative to professionals and other users.</p>
                <ul>
                    <li><strong>Sending Feedback</strong> Go to <a href="https://forms.gle/zdpUEc1exkUhDdfp7" target="_blank" style="color: #f2c94c;">this Link</a> to submit relevant feedback for our website.
                    </li>
                </ul>
            </div>
        </div>
    </div>
`;

// Inject the HTML into the page as soon as it loads
document.addEventListener('DOMContentLoaded', () => {
    document.body.insertAdjacentHTML('beforeend', helpModalHTML);
});