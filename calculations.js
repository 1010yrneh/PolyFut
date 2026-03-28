/* calculations.js - TRUE HYBRID VALUATION ENGINE (WITH DYNAMIC SHADOW MULTIPLIERS) */

// 1. Exact Markov values from CSV
const MARKOV_COEFFS = {
    'FW': {
        'Shot Taken': 0.1108, 'Key Pass': 0.1108, 'Carry into Box': 0.0519, 'Pass into Box': 0.0519, 'Cross into Box': 0.0545, 
        'Progression (Pass)': 0.0042, 'Progression (Carry)': 0.0042, 'Dispossessed': -0.0053, 'Dribble (Beat Man)': 0.0, 
        'Block': 0.0001, 'High Press Win': 0.0293, 'Midfield Tackle': 0.00147, 'Ball Recovery': 0.00098, 
        'Aerial Duel Won': 0.00098, 'Deep Tackle': 0.00049, 'Foul Committed': -0.00049, 'Interception': 0.00117,
        'Defensive Error': -0.03459
    },
    'MF': {
        'Shot Taken': 0.1108, 'Key Pass': 0.1108, 'Pass into Box': 0.0519, 'Carry into Box': 0.0519, 'Cross into Box': 0.0545, 
        'Progression (Carry)': 0.0042, 'Progression (Pass)': 0.0042, 'Dispossessed': -0.0053, 'Dribble (Beat Man)': 0.0, 
        'Block': 0.00041, 'High Press Win': 0.01627, 'Midfield Tackle': 0.00203, 'Deep Tackle': 0.00122, 
        'Aerial Duel Won': 0.00081, 'Ball Recovery': 0.00065, 'Foul Committed': -0.00065, 'Interception': 0.00163,
        'Defensive Error': -0.03459
    },
    'DF': {
        'Shot Taken': 0.1108, 'Key Pass': 0.1108, 'Carry into Box': 0.0519, 'Cross into Box': 0.0545, 'Pass into Box': 0.0519, 
        'Progression (Pass)': 0.005, 'Progression (Carry)': 0.005, 'Dribble (Beat Man)': 0.0, 'Dispossessed': -0.008, 
        'Block': 0.01729, 'Deep Tackle': 0.00778, 'Interception': 0.00182, 'Ball Recovery': 0.00104, 
        'Midfield Tackle': 0.00078, 'Aerial Duel Won': 0.00062, 'Foul Committed': -0.00104, 'Defensive Error': -0.03459
    }
};

// 2. Exact Ridge values from CSV
const RIDGE_COEFFS = {
    'FW': {
        'Shot Taken': 0.1349, 'Key Pass': -0.028, 'Carry into Box': 0.0396, 'Pass into Box': 0.0036, 'Cross into Box': -0.0464, 
        'Progression (Pass)': -0.0056, 'Progression (Carry)': -0.0193, 'Dispossessed': -0.0077, 'Dribble (Beat Man)': -0.0385, 
        'Block': 0.20911, 'High Press Win': 0.0, 'Midfield Tackle': 0.0, 'Ball Recovery': 0.0, 'Aerial Duel Won': 0.0, 
        'Deep Tackle': 0.0, 'Foul Committed': 0.0, 'Interception': -0.08213, 'Defensive Error': -0.06951
    },
    'MF': {
        'Shot Taken': 0.1144, 'Key Pass': -0.0052, 'Pass into Box': 0.0155, 'Carry into Box': -0.0122, 'Cross into Box': -0.0406, 
        'Progression (Carry)': 0.0131, 'Progression (Pass)': -0.0099, 'Dispossessed': 0.0041, 'Dribble (Beat Man)': -0.0356, 
        'Block': 0.27989, 'High Press Win': 0.0, 'Midfield Tackle': 0.0, 'Deep Tackle': 0.0, 'Aerial Duel Won': 0.0, 
        'Ball Recovery': 0.0, 'Foul Committed': 0.0, 'Interception': -0.1938, 'Defensive Error': -0.06951
    },
    'DF': {
        'Shot Taken': 0.1007, 'Key Pass': 0.0017, 'Carry into Box': 0.0212, 'Cross into Box': 0.006, 'Pass into Box': -0.0373, 
        'Progression (Pass)': 0.0006, 'Progression (Carry)': -0.0061, 'Dribble (Beat Man)': -0.0006, 'Dispossessed': 0.0198, 
        'Block': -0.00644, 'Deep Tackle': 0.0, 'Interception': -0.00328, 'Ball Recovery': 0.0, 'Midfield Tackle': 0.0, 
        'Aerial Duel Won': 0.0, 'Foul Committed': 0.0, 'Defensive Error': -0.06951
    }
};

// 3. Exact Hybrid values from CSV 
const HYBRID_COEFFS = {
    'FW': {
        'Shot Taken': 0.1378, 'Key Pass': 0.1052, 'Carry into Box': 0.0598, 'Pass into Box': 0.0526, 'Cross into Box': 0.0452, 
        'Progression (Pass)': 0.003, 'Progression (Carry)': 0.0003, 'Dispossessed': -0.0069, 'Dribble (Beat Man)': -0.0077, 
        'Block': 0.0419, 'High Press Win': 0.02344, 'Midfield Tackle': 0.00117, 'Ball Recovery': 0.00078, 
        'Aerial Duel Won': 0.00078, 'Deep Tackle': 0.00039, 'Foul Committed': -0.00039, 'Interception': -0.01549,
        'Defensive Error': -0.03983
    },
    'MF': {
        'Shot Taken': 0.1337, 'Key Pass': 0.1097, 'Pass into Box': 0.055, 'Carry into Box': 0.0495, 'Cross into Box': 0.0464, 
        'Progression (Carry)': 0.0068, 'Progression (Pass)': 0.0022, 'Dispossessed': -0.0045, 'Dribble (Beat Man)': -0.0071, 
        'Block': 0.08425, 'High Press Win': 0.01139, 'Midfield Tackle': 0.00142, 'Deep Tackle': 0.00085, 
        'Aerial Duel Won': 0.00057, 'Ball Recovery': 0.00046, 'Foul Committed': -0.00046, 'Interception': -0.057,
        'Defensive Error': -0.03983
    },
    'DF': {
        'Shot Taken': 0.1309, 'Key Pass': 0.1111, 'Carry into Box': 0.0562, 'Cross into Box': 0.0557, 'Pass into Box': 0.0445, 
        'Progression (Pass)': 0.0051, 'Progression (Carry)': 0.0038, 'Dribble (Beat Man)': -0.0001, 'Dispossessed': -0.004, 
        'Block': 0.01373, 'Deep Tackle': 0.00662, 'Interception': 0.00105, 'Ball Recovery': 0.00088, 
        'Midfield Tackle': 0.00066, 'Aerial Duel Won': 0.00053, 'Foul Committed': -0.00088, 'Defensive Error': -0.03983
    }
};

const OFFENSIVE_ACTIONS = ['Shot Taken', 'Key Pass', 'Carry into Box', 'Pass into Box', 'Cross into Box', 'Progression (Pass)', 'Progression (Carry)', 'Dribble (Beat Man)', 'Goal', 'Assist'];
const DEFENSIVE_ACTIONS = ['High Press Win', 'Midfield Tackle', 'Deep Tackle', 'Interception', 'Block', 'Ball Recovery', 'Aerial Duel Won'];
const RISK_ACTIONS = ['Dispossessed', 'Defensive Error', 'Foul Committed'];

// --- THE SHADOW MULTIPLIER ENGINE ---
function applyShadowValues(coeffs) {
    let scaled = { ...coeffs };
    
    // The Multiplier: |Defensive Error| / |Shot Taken|
    let baseError = Math.abs(scaled['Defensive Error'] || 0.03459);
    let baseShot = Math.abs(scaled['Shot Taken'] || 0.1108);
    let shadowMultiplier = baseShot === 0 ? 0 : baseError / baseShot;
    
    // REALISTIC MAPPING
    const SHADOW_MAP = {
        'Block': 'Shot Taken',                  
        'High Press Win': 'Carry into Box',     
        'Deep Tackle': 'Pass into Box',         
        'Interception': 'Pass into Box',        // Upgraded to give proper credit for cutting passing lanes
        'Midfield Tackle': 'Progression (Carry)',
        'Ball Recovery': 'Progression (Pass)',  
        'Aerial Duel Won': 'Cross into Box'     
    };

    // ONLY boost the positive defensive actions. 
    for (let defAction in SHADOW_MAP) {
        let offAction = SHADOW_MAP[defAction];
        
        // 1. Calculate the shadow value stolen from the offense
        let shadowValue = Math.abs(scaled[offAction] || 0) * shadowMultiplier;
        
        // 2. THE FIX: Erase Possession Bias. If the raw Ridge correlation punished 
        // this defensive action, we flip it to a positive reward using Math.abs()
        let rawBase = scaled[defAction] || 0;
        let correctedBase = Math.abs(rawBase); 
        
        // 3. Add them together for the true defensive value
        scaled[defAction] = correctedBase + shadowValue;
    }

    return scaled;
}

function calculatePerformance(matchStats, currentScore, duration, excludedRanges, position) {
    let offMarkov = 0, offRidge = 0;
    let defMarkov = 0, defRidge = 0;
    let netScore = 0;
    
    let chartData = [{x: 0, y: 0}];
    let runningTotal = 0;

    // Fetch Base Coefficients
    const baseMarkov = MARKOV_COEFFS[position] || MARKOV_COEFFS['MF'];
    const baseRidge = RIDGE_COEFFS[position] || RIDGE_COEFFS['MF'];
    const baseHybrid = HYBRID_COEFFS[position] || HYBRID_COEFFS['MF'];

    // Dynamically apply the Shadow Values
    const pMarkov = applyShadowValues(baseMarkov);
    const pRidge = applyShadowValues(baseRidge);
    const pHybrid = applyShadowValues(baseHybrid);

    matchStats.forEach(stat => {
        let isExcluded = excludedRanges.some(range => stat.seconds >= range.start && stat.seconds <= range.end);
        if (isExcluded) return;

        // Fetch the NEW SCALED values
// Fetch the NEW SCALED values
        let rawMVal = pMarkov[stat.action] || 0;
        let rawRVal = pRidge[stat.action] || 0;
        let rawHVal = pHybrid[stat.action] || 0;

        // PURE DECIMALS (Literal xG and Win%)
        let mVal = rawMVal;
        let rVal = rawRVal;
        let hVal = rawHVal;

        // The running total and chart use the TRUE Hybrid score
        runningTotal += hVal;
        netScore += hVal;

        // The breakdowns use the separated M/R values
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

        // Use 3 decimal places for the chart so lines don't look flat
        chartData.push({ x: stat.seconds, y: runningTotal.toFixed(3) });
    });

    if (duration > 0) {
        chartData.push({ x: duration, y: runningTotal.toFixed(3) });
    }

    // Return everything rounded to 3 decimal places for the UI
    return {
        netScore: netScore.toFixed(3),
        offMarkov: offMarkov.toFixed(3),
        offRidge: offRidge.toFixed(3),
        defMarkov: defMarkov.toFixed(3),
        defRidge: defRidge.toFixed(3),
        chartData: chartData,
        
        // Pass the SCALED values to the AI Agent so it fully understands the boosted defense
        coeffMarkov: pMarkov,
        coeffRidge: pRidge,
        coeffHybrid: pHybrid
    };
}