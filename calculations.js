/* calculations.js — PolyFut hybrid valuation (Threat Points; source v3.3.2) */
/* Unit: 1 TP = 1% of a goal. Goal = 100 TP. Assist = pure research × 100. */
/* Creation rarity priors on FW/MF/DF (v3.3.2). Shadow multiplier removed. */
/* Goal auto-suppresses a preceding Shot Taken within 12s (no double-count). */

function _pfDecode(blob) {
    const key = [80, 111, 108, 121, 70, 117, 116, 51]; // PolyFut3
    const bin = atob(blob);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i) ^ key[i % key.length];
    return JSON.parse(new TextDecoder().decode(out));
}

const _PF = _pfDecode(
    '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'
);

const MARKOV_COEFFS = _PF.M;
const RIDGE_COEFFS = _PF.R;
const HYBRID_COEFFS = _PF.H;
const SCORE_UNIT = 'TP';
const GOAL_SHOT_SUPPRESS_WINDOW_SEC = 12;

const OFFENSIVE_ACTIONS = ['Shot Taken', 'Key Pass', 'Carry into Box', 'Pass into Box', 'Cross into Box', 'Progression (Pass)', 'Progression (Carry)', 'Dribble (Beat Man)', 'Goal', 'Assist'];
const DEFENSIVE_ACTIONS = ['High Press Win', 'Midfield Tackle', 'Deep Tackle', 'Interception', 'Block', 'Ball Recovery', 'Aerial Duel Won'];
const RISK_ACTIONS = ['Dispossessed', 'Defensive Error', 'Foul Committed'];

function coeffRelativeLabels(coeffs) {
    const entries = Object.entries(coeffs || {}).filter(([, v]) => typeof v === 'number');
    if (!entries.length) return {};
    const absVals = entries.map(([, v]) => Math.abs(v)).sort((a, b) => a - b);
    const q = (p) => absVals[Math.min(absVals.length - 1, Math.floor(p * (absVals.length - 1)))];
    const hi = q(0.66), mid = q(0.33);
    const out = {};
    for (const [k, v] of entries) {
        const a = Math.abs(v);
        const mag = a >= hi ? 'high' : a >= mid ? 'medium' : 'low';
        out[k] = (v < 0 ? '-' : '+') + mag;
    }
    return out;
}

function markSuppressedShots(matchStats) {
    const suppressed = new Set();
    for (let i = 0; i < matchStats.length; i++) {
        if (matchStats[i].action !== 'Goal') continue;
        const gSec = matchStats[i].seconds;
        let best = -1;
        let bestDt = Infinity;
        for (let j = 0; j < matchStats.length; j++) {
            if (matchStats[j].action !== 'Shot Taken') continue;
            if (suppressed.has(j)) continue;
            const dt = gSec - matchStats[j].seconds;
            if (dt >= 0 && dt <= GOAL_SHOT_SUPPRESS_WINDOW_SEC && dt < bestDt) {
                bestDt = dt;
                best = j;
            }
        }
        if (best >= 0) suppressed.add(best);
    }
    return suppressed;
}

function calculatePerformance(matchStats, currentScore, duration, excludedRanges, position) {
    let offMarkov = 0, offRidge = 0;
    let defMarkov = 0, defRidge = 0;
    let netScore = 0;

    let chartData = [{x: 0, y: 0}];
    let runningTotal = 0;

    const pMarkov = MARKOV_COEFFS[position] || MARKOV_COEFFS['MF'];
    const pRidge = RIDGE_COEFFS[position] || RIDGE_COEFFS['MF'];
    const pHybrid = HYBRID_COEFFS[position] || HYBRID_COEFFS['MF'];
    const suppressedShots = markSuppressedShots(matchStats);

    matchStats.forEach((stat, idx) => {
        let isExcluded = excludedRanges.some(range => stat.seconds >= range.start && stat.seconds <= range.end);
        if (isExcluded) return;
        if (suppressedShots.has(idx)) return;

        let mVal = pMarkov[stat.action] || 0;
        let rVal = pRidge[stat.action] || 0;
        let hVal = pHybrid[stat.action] || 0;

        runningTotal += hVal;
        netScore += hVal;

        if (OFFENSIVE_ACTIONS.includes(stat.action)) {
            offMarkov += mVal;
            offRidge += rVal;
        } else if (DEFENSIVE_ACTIONS.includes(stat.action)) {
            defMarkov += mVal;
            defRidge += rVal;
        } else if (RISK_ACTIONS.includes(stat.action)) {
            if (stat.action === 'Dispossessed') {
                offMarkov += mVal; offRidge += rVal;
            } else {
                defMarkov += mVal; defRidge += rVal;
            }
        }

        chartData.push({ x: stat.seconds, y: Math.round(runningTotal * 10) / 10 });
    });

    if (duration > 0) {
        chartData.push({ x: duration, y: Math.round(runningTotal * 10) / 10 });
    }

    return {
        netScore: netScore.toFixed(1),
        offMarkov: offMarkov.toFixed(1),
        offRidge: offRidge.toFixed(1),
        defMarkov: defMarkov.toFixed(1),
        defRidge: defRidge.toFixed(1),
        chartData: chartData,
        unit: SCORE_UNIT,
        coeffMarkov: coeffRelativeLabels(pMarkov),
        coeffRidge: coeffRelativeLabels(pRidge),
        coeffHybrid: coeffRelativeLabels(pHybrid)
    };
}
