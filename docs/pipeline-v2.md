# PolyFut CV Pipeline v2 — Ball-Anchored, Single-Player

**Status:** Proposed (design) · **Author:** Henry Chien · **Date:** 2026-07-04

This document describes a redesign of the Level 1 `polyfut_video` pipeline. It is a
design proposal, not yet implemented — the current shipped pipeline is the
team-possession model described in the repo README.

---

## 1. Motivation

The current pipeline detects **team possession** and marks every window where *anyone*
on your team has the ball. Two problems follow:

1. **Over-labelling.** A team-possession model surfaces every teammate's touch, so a
   full match produces far more hotspots than a single player cares about. If the labels
   are wrong or too broad, the 3–5 hours of CPU processing is effectively wasted.
2. **Wrong scope for the product.** PolyFut is for a player analysing *their own*
   performance. You only care about *your* touches, not the whole team's.

**v2 reframes the problem around a single player and around the ball.** A single outfield
player only touches the ball ~30–80 times a match, which collapses to a very reviewable
set of hotspots.

---

## 2. Design principles

1. **The ball is the anchor, not the player.** There is exactly one ball, so tracking it
   needs no re-identification. Everything hangs off the ball trajectory. Tracking one
   *player* across 90 minutes (long-term re-ID, identical kits) is the genuinely hard,
   semi-unsolved problem — we avoid depending on it.
2. **Compute is sparse.** Only the ball is analysed every frame. The expensive work —
   player detection, identity, motion reasoning — runs only at the few hundred moments
   the ball is actually contacted.
3. **CV generates candidates; the human decides.** The pipeline is tuned for **high
   recall**. The review step supplies **precision** on the one thing CV cannot reliably
   resolve on amateur footage: you vs. an identical-kit teammate. This makes an
   over-generating run *robust* rather than *wasted* — you always get a correct result.

---

## 3. Architecture: continuous vs sparse

```
CONTINUOUS  (~40k analysed frames)              SPARSE  (~150–300 contact moments)
────────────────────────────────────           ──────────────────────────────────
 decode → shot/deadtime filter                   player detection near ball
        → BALL detection ───► ball trajectory ─► per-contact color filter (your team?)
                                   │             → target score = appearance × orbital
                                   ▼             → human me / not-me review (clips)
                          contact candidates ───► ±2s window → merged hotspots
                          (ball kinematics)
```

The single most important change: **full-frame player detection moves from every frame
(left column) to only the contact candidates (right column).**

---

## 4. Stage-by-stage

Tags: **[reuse]** unchanged from v1 · **[changed]** modified · **[new]** · **[drop]** removed.

### Stage 0 — Multi-sample seed · [changed]
Instead of one tap on frame 0 (the target may not be visible at kickoff, and a single
crop is a brittle appearance model):

1. Run a quick decode + **shot-filter** pass so sampling frames are guaranteed to be live
   play (no replays/graphics/crowd shots).
2. Present **3–4 frames spread across the match timeline** (e.g. ~10%, 35%, 60%, 85%).
   The user **taps their player** in each frame where they are visible, and **skips** any
   frame where they are not.
3. From those crops, build:
   - a **target appearance gallery** (multiple poses / angles / lighting) for Stage 7, and
   - a robust **median kit color** for the Stage 6 filter.

Rules of thumb: up to 4 samples, minimum 2; warn on 1 (weak gallery → larger montage).
Spread across the timeline captures appearance drift (lighting, sleeves, mud, bibs) so a
minute-80 contact still matches.

### Stage 1 — Decode · [reuse]
Downscale to 640, frame sampling. Unchanged (`decode.py`).

### Stage 2 — Shot filter + deadtime · [reuse]
Drop replays, graphics, and dead time before the continuous pass shrinks the frame count
(`shot_filter.py`, `deadtime_filter.py`).

### Stage 3 — Continuous ball tracking · [changed] — now the main continuous cost
Run **only the soccer-specific ball model** every analysed frame. Search a region of
interest around the last known ball position (cheaper + higher effective resolution on a
tiny object). Interpolate across misses (reuse `ball_smooth.py` / `ball_hold_frames`).
**Player detection does NOT run here.** Output: ball trajectory `(x, y, t)` with
confidence and gap flags.

### Stage 4 — Contact-candidate detection from ball kinematics · [new]
From the trajectory *alone*, flag "someone touched the ball" frames: sharp direction
changes, decelerations/stops, speed spikes (a kick). Pure signal processing, very cheap.
Output: a few hundred candidate contact timestamps. No player information needed yet.

*Risk:* under ball occlusion the trajectory has gaps that can fake or hide inflections. A
fallback proximity scan in dense regions may be needed (see §8).

### Stage 5 — Sparse player detection at candidates · [changed] — the big speed win
**Only** at candidate frames (plus a few frames either side), run player detection on a
crop around the ball. Identify the player(s) in contact (nearest / overlapping bbox).
Replaces v1's per-frame full-frame player detection.

### Stage 6 — Per-contact team color filter · [changed] — replaces DBSCAN clustering
At each candidate, sample the contacting player's jersey over the ±few frames, take a
**median hue in HSV/Lab**, and compare to the seed color. Keep **your-team** contacts,
drop opponents. This roughly **halves** the candidate set and is what keeps the montage
your-team-only.

This *replaces* the heavy whole-match team clustering (`team_classify.py` DBSCAN) but
keeps robustness by averaging a few frames per contact. Note: the color filter is cheap
either way — the speed win is Stage 5, not this.

### Stage 7 — Target scoring: appearance × motion (sequential) · [new]
For each surviving your-team contact, produce a **confidence** (not a hard yes/no) used to
rank the review montage and to auto-accept / auto-hide the extremes.

Confidence combines two complementary signals:

- **Appearance match** — similarity of the contacting player to the Stage 0 **gallery**
  (color histogram + optional ReID / SigLIP embedding + jersey-number OCR if legible).
  Gallery matching (min/mean distance over several reference crops) beats single-crop
  matching on same-kit teammates.
- **Orbital motion prior** — a bounded region ("orbital") predicting where the target can
  be, anchored on the last high-confidence sighting, with a radius that grows with the
  time gap. A contact inside the orbital is boosted; one that would require the player to
  teleport is suppressed. This is near-zero compute (geometry on positions we already
  have) and is strongest exactly where appearance is weakest — temporally-clustered
  contacts (dribbles, give-and-goes) where teammates bunch around the ball. It can also
  break the hardest tie: two same-kit teammates at one contact, only one motion-consistent.

```
score = appearance_match(gallery) × orbital_prior(Δt, distance)
```

Practically this promotes scoring from *independent per-contact* to *sequential*: link
nearby contacts into short target **tracklets** and score them jointly.

**Orbital caveats (see also §8):**
- *Coordinate frame.* An orbital is clean in real-world meters but unreliable in raw
  pixels on a panning/zooming camera. v1 requires at least **camera-motion compensation**
  (cheap global optical-flow/feature estimate); the clean version wants **pitch
  homography** (optional keypoint model). The orbital is the strongest argument for
  eventually adding homography.
- *Anchor drift.* The orbital predicts from a *known* position; a wrong anchor propagates.
  Only (re)anchor from high-confidence appearance matches; let the radius grow while
  unanchored.
- *Long gaps.* Sparse contacts mean the orbital can grow to cover the pitch → no signal.
  It then degrades gracefully to "no prior."
- *v1 safety.* Keep the orbital **conservative**: generous radius, camera-motion
  compensated, anchor only on strong matches, and allow it to **boost / tie-break only —
  never hard-reject** — so a shaky prior can never cause a silent false negative.

### Stage 8 — Human review montage · [new] — core UX, the "never wasted" property
Present surviving candidates as short **1–2 s clips** (cropped/zoomed to the contact),
**ranked by Stage 7 confidence**, with the clear extremes auto-accepted / auto-hidden. The
user taps **me / not-me**. Use clips, not stills — motion and context make "me vs. #7" far
easier for a human than a frozen freeze-frame. This is where high-recall CV becomes
high-precision output.

### Stage 9 — Hotspot assembly · [reuse, tuned]
For each confirmed "me" contact, take **±2 s** → window; **merge** windows within the gap
threshold (handles dribbles / quick sequences), pad, apply min-zone. Emit the final
hotspot timeline into the workspace + logging UI. Reuses the existing `hotspot_*` config
(`hotspot_pad_before_sec`, `hotspot_gap_merge_sec`, `hotspot_min_zone_sec`), fed cleaner
single-player events.

---

## 5. What v2 drops relative to v1

- **Whole-match team classification** (DBSCAN over accumulated crops, `team_classify.py`)
  → replaced by the cheap per-contact color check (Stage 6).
- **Continuous full-frame player detection** → moved to sparse candidate frames (Stage 5).
- **Team-possession windowing** (`possession.py` team logic) → replaced by single-player
  contact events (Stages 4–9).

---

## 6. Compute profile

- **Continuous cost:** ball detection only — your hardest object, but a single small
  model per frame. Estimated **~1.5–2× faster end-to-end** than v1's player+ball every
  frame. *Not* 10×: ball-every-frame is the floor.
- **Sparse cost:** player detection + scoring at ~150–300 moments — a small fraction of
  the continuous cost.
- **Orbital:** negligible (geometry on existing positions). Optional camera-motion
  estimate adds a small, bounded cost.
- **Human cost shifts up:** ~150–300 clip judgments, trimmed by Stage 7 ranking /
  auto-accept. This is the deliberate trade: slower-per-human, but the run can never
  produce garbage.

*All numbers are estimates pending real footage; see §8.*

---

## 7. Reliability profile

Reliability is **asymmetric**, which is favourable:

- **False positives** (ball near a player, no real touch, or a teammate mis-scored high):
  cheap — rejected in the montage. Harmless.
- **False negatives** (a real touch never surfaced): the only dangerous, **silent**
  failure — it never appears in the montage, so it can't be recovered. Every miss traces
  back to the **ball not being detected** at that moment. **Ball-model quality is the
  reliability ceiling** — a soccer-specific ball model + trajectory interpolation is
  non-negotiable.
- **Identity ceiling** (you vs. same-kit teammate) is unchanged in principle, but moved
  from *CV guessing* to *you deciding from a 2 s clip*, which is the right place for it.
  Orbital motion-continuity + gallery appearance narrow how often the human is needed.

---

## 8. Open questions / risks

1. **Footage type (biggest unknown).** Broadcast-style (camera follows the ball — the
   tailwind that puts touches on-screen) vs. a fixed wide camera (whole pitch, tiny
   players — hurts ball recall *and* montage legibility). This materially changes every
   estimate above. *Needed to turn ranges into real numbers.*
2. **Kinematic candidate detection under occlusion.** Ball gaps in crowds can fake/hide
   inflections; may need a fallback proximity scan in dense regions.
3. **Orbital coordinate frame.** Reliable only with camera-motion compensation or pitch
   homography; raw-pixel orbitals are shaky.
4. **Montage size / human load.** Depends on Stage 7 ranking + auto-accept working well;
   a weak gallery (only 1 seed sample) inflates it.
5. **Domain gap.** Public soccer models are trained on pro broadcast; amateur/youth
   footage differs. Fine-tuning on a few hundred labeled frames of real footage is the
   real accuracy ceiling.

---

## 9. Model / repo dependencies

| Stage | Need | Candidate |
|-------|------|-----------|
| 3 — ball | Soccer-specific ball detector (replaces COCO ball) | Roboflow `sports` ball model / fine-tune |
| 5 — players | Soccer player/keeper/ref/ball detector | Roboflow `football-players-detection` |
| 7 — identity | Appearance embedding + optional number OCR | torchreid / SigLIP; methods ref: SoccerNet tracking / re-ID |
| 7 — orbital (clean) | Pitch homography (optional) | Roboflow `sports` pitch-keypoint model |

All detectors are Ultralytics-compatible, so the `yolo_weights` swap stays drop-in
(`polyfut_video/config.py`, `polyfut_video/pipeline/detection.py`).

---

## 10. Config deltas (sketch)

Relative to `polyfut_video/config.py`:

- **Add:** soccer `ball_weights` + `player_weights` (split from single `yolo_weights`);
  `seed_sample_count` (default 4, min 2); contact-kinematics thresholds
  (min speed change / stop detection); orbital params (base radius, growth rate per
  second, camera-motion on/off); Stage 7 auto-accept / auto-hide thresholds.
- **Repurpose:** `ball_hold_frames`, `ball_max_jump_px` → Stage 3 interpolation.
- **Keep:** `hotspot_pad_*`, `hotspot_gap_merge_sec`, `hotspot_min_zone_sec` → Stage 9
  (change pad to ±2 s).
- **Retire:** `dbscan_*`, `min_cluster_size`, `team_crops_per_track`, team-possession
  thresholds.
