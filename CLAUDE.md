# CLAUDE.md

Guidance for agents working in this repo. This file focuses on the **object
classification / player identification** problem in the v2 pipeline — the
source of the "wrong team, referees, and random sideline objects get tagged"
symptoms. Read this before changing anything in Stages 5–8.

For the overall pipeline design, see `docs/pipeline-v2.md`.

---

## 0. TL;DR for the two work directions

1. **Differentiate the other team** → build a **per-match team classifier** in
   the pipeline (assign every detected player to `team_a` / `team_b` /
   `official` / `unknown` from kit colour, re-derived each match). A pre-trained
   detection-model class for "the other team" **cannot work** — kit colours
   change every match. See §3.
2. **Real identification (you vs. same-kit teammate)** → the ceiling is real and
   colour cannot cross it. Priorities, in order:
   - **(A) Motion / tracking continuity — the primary signal.** In this
     ball-anchored context it is the strongest lever we have. Strengthen the
     orbital/tracklet mechanism so identity is *carried* between confident
     sightings rather than re-decided per touch on appearance. See §4.1.
   - **(B) A ReID embedding** to replace the colour histogram as a secondary,
     more-discriminative appearance signal. See §4.2.
   - **(C) Honesty about the ceiling.** Same-kit ID is semi-unsolved on this
     footage; the human review step is the intended precision layer. See §4.3.

---

## 1. The symptoms and their real causes

Observed on the review screen: players from the other team, referees, and
isolated sideline objects get tagged as touch candidates.

| Symptom | Actual cause | Where |
|---|---|---|
| Referee tagged as a player | The model **mis**classifies the ref as class `player`; we only ever request that one class, so we can't use the ref class to exclude them. | `player_detector.py` requests `classes=[player_class_id]` only |
| Goalkeeper touches never appear | Keeper is a **separate class** (`1`) that is never requested → structurally invisible, not misattributed. | `ball_model.py` class map |
| Sideline / bench / near-graphic object tagged | "Nearest player box to the ball" wins with **no check that it's actually a contesting player** (only a max-distance gate + a ball-shape reject). | `player_contacts.py::nearest_player` |
| Wrong-team touch survives | The team gate keeps a wide **"undecided" colour band** (60–115) rather than dropping it — deliberate recall-safety. | `player_contacts.py::enrich_contact` |
| You can't be told apart from a teammate | "Appearance" is an **HSV colour histogram** — the same kind of signal as the team gate. It cannot separate same-kit players by construction. | `appearance.py::HistogramAppearance` |

**None of these stages ever hard-rejects.** By design every signal is soft
(team keeps "undecided", appearance floors at `appearance_default=0.5`, orbital
floors at `orbital_floor=0.5`). The pipeline over-includes on purpose and pushes
ambiguity to the human review montage. So "the review screen shows junk" is
partly the system *working as designed on genuinely low-confidence detections* —
but the specific cases in the table above are fixable and worth fixing.

Aggravating factor, always: **most real footage here is 640×360** (players
~15px). This is near the discrimination ceiling for colour and for OCR. The app
already warns about this (`pfShowPipelineWarnings`); do not "fix" a resolution
limitation by tightening a threshold until real touches get dropped.

---

## 2. How identification works today (the map)

```
detect (class=player only)          player_detector.py   — classes=[player_class_id]
  → nearest box to ball              player_contacts.py::nearest_player
  → grass-masked jersey HSV          color.py::jersey_hsv
  → 3-way team gate                  player_contacts.py::enrich_contact
        dist ≤ 60  → my team
        dist > 115 → other team → DROP
        else       → UNDECIDED → keep
  → appearance = HS histogram        appearance.py::HistogramAppearance
  → confidence = appearance × orbital  scoring.py::score_contacts
  → review montage + grouping        montage.py, grouping.py (review-UI only)
```

Key config (`polyfut_v2/config.py`): `team_color_max_dist=60`,
`contact_other_team_dist=115`, `team_filter_enabled=False` (the aggressive gate
is off by default), `contact_max_player_dist_px=80`, `appearance_default=0.5`,
`orbital_anchor_min=0.6`, `orbital_base_px=80`, `orbital_growth_px_s=60`,
`orbital_max_gap_sec=8`, `orbital_floor=0.5`, `tracklet_max_gap_sec=3`.

**What already exists to build on (do not reinvent):**
- Per-match 2-team **k-means** on torso colours, used for the "pick your team"
  swatches: `team_preview.py::_kmeans_two_kits` (grass-masked).
- Review-time team + look-alike **grouping**: `grouping.py::assign_player_groups`
  emits `kit_group` / `is_other_team` / `appearance_group` per montage item.
  **But it only powers review-UI decision propagation** (called at
  `app_service.py:~395`) — it is *not* a gate during detection/scoring.
- Grass-masked colour primitives: `color.py` (`jersey_hsv`, `hsv_distance`,
  `hex_to_hsv`, `grass_fraction`). The grass mask is load-bearing — a torso crop
  is mostly grass at 640×360 and reads green without it.

---

## 3. Direction A — a per-match "other team" class

### Why NOT a detection-model class
Kit colours are different every match (and both teams pick fresh kits per
fixture). A model class named "other team" would have to mean a *specific
colour*, which does not generalize. **Do not** propose retraining the detector to
output team. Team membership is inherently a **per-match clustering** problem.

### The approach
Formalize a first-class, per-match team label for **every detected player**, not
just a soft distance to the seed kit:

1. **Establish the two team centroids once per match**, from a robust sample of
   grass-masked torso colours across several early, populated frames. This is
   essentially what `_kmeans_two_kits` already does for the picker — promote it
   to a pipeline-wide "team model" (two HSV centroids) computed once and reused.
2. **Classify each detected player** at contact time into
   `team_a` / `team_b` / `official` / `unknown`:
   - `team_a` / `team_b`: nearest centroid within a confidence margin.
   - `official`: **use the model's existing classes** — request `referee` (3)
     and `goalkeeper` (1) alongside `player` (2) so a correctly-classified ref is
     *excluded by class*, and keeper touches become *detectable* instead of
     invisible. (Cheap, high-value — see quick win below.)
   - `unknown`: colour unreadable (too small / too much grass) → keep for recall,
     exactly as today's "undecided" band does.
3. **The user's team is one of the two centroids** (they already pick it on the
   team screen; `cvMyTeam.hex` flows to the backend). "Other team" = the *other*
   centroid — a positive assignment, not just "far from my kit". This is stronger
   than the current one-sided `dist > 115` test because it uses *both* clusters:
   a touch near the opponent centroid is opponent even if it's also not-that-far
   from yours (the ambiguous 60–115 band shrinks).

### Quick win (do this first — small, isolated, testable)
Request `goalkeeper` and `referee` from the model and handle them explicitly:
- Referee (class 3) detections → never a touch candidate.
- Goalkeeper (class 1) → a valid player (keepers touch the ball).
Today `player_detector.py` hard-codes `classes=[player_class_id]`. This is the
single cheapest reduction in referee mixups and the only way keeper touches ever
get seen. Verify against real footage that the model's class 3 actually fires on
refs at 640×360 before relying on it — if recall on the ref class is poor, fall
back to the colour/position heuristics.

### Wiring note
The team classifier should become a **gate in `enrich_contact`** (drop confirmed
opponents) and a **signal into scoring**, not just the review-time
`grouping.py`. Keep the recall-safety rule: only *confirmed* opponents drop;
`unknown` always survives to review.

---

## 4. Direction B — real identification (you vs. same-kit teammate)

### 4.1 Motion / tracking continuity — PRIMARY, build this first
In a ball-anchored pipeline this is the strongest identity signal available and
should be treated as the backbone of identity, above appearance.

- The mechanism already exists but is deliberately weak: `scoring.py` has an
  **orbital prior** (a growing search radius from the last high-confidence
  sighting) and **tracklets** (contacts within `tracklet_max_gap_sec=3` link).
  It is currently a *soft tie-breaker* — `orbital_floor=0.5`, and it "never hard
  rejects" so a bad prior can't cause a false negative.
- The direction: make continuity **carry identity between confident sightings**
  rather than re-deciding each touch from appearance. Concretely worth exploring:
  - Continuous (not just per-contact) tracking of the anchored player between
    touches, so the identity thread survives gaps the current 3s tracklet cutoff
    breaks. (The seed-clip tracker in `seed_clips.py::_track` — colour-locked
    nearest-centroid with a merge/rejoin pass — is the closest existing pattern
    to reuse.)
  - **Camera-motion compensation.** `scoring.py` has a `transform` hook
    (identity by default) explicitly for pitch homography / camera-pan
    compensation. Orbital reasoning in raw pixel space is wrong during pans;
    this hook is where that gets fixed and would materially strengthen the
    motion prior. This is likely the highest-leverage single improvement.
  - Let a strong motion chain *override* a weak/ambiguous appearance read, not
    just tie-break it — invert today's "appearance leads, motion nudges".
- Constraint: motion continuity **drifts over long gaps** and through occlusions
  and cuts; it cannot be the *only* signal. Pair it with the human review, and
  degrade gracefully to "neutral" past `orbital_max_gap_sec` (as it already
  does).

### 4.2 ReID embedding — SECONDARY appearance signal
Replace/augment `HistogramAppearance` (an H-S colour histogram) with a learned
torso/body embedding (OSNet-style ReID or similar) behind the existing
`AppearanceModel` protocol in `appearance.py` — it's designed to be swapped.
More discriminative than colour, but **same-kit teammates remain genuinely hard**
and it adds a model dependency + compute. Treat as an upgrade to the appearance
*half* of `confidence = appearance × orbital`, not a silver bullet.

### 4.3 Known dead-end — do not repeat
A **region-based body descriptor** (head/torso/legs HSV histograms, kit region
de-weighted) was tried as a re-ID feature and **failed**: on two well-separated
same-kit players it gave within=0.962, cross=0.891 → **separation 0.070**, worse
than the plain torso histogram's 0.153. Richer *hand-crafted colour* descriptors
do not solve same-kit ID. A real learned embedding (§4.2) is a different thing;
another colour-feature variant is not worth trying.

### 4.4 The honest ceiling
Two outfield players in the same kit at ~15px are close to indistinguishable to
any appearance signal. This is *why* the product design (`docs/pipeline-v2.md`
§2) anchors on the ball (one object, no re-ID) and uses **human review** as the
precision layer for identity. Frame same-kit ID as semi-unsolved: pursue motion
continuity hard, add a ReID embedding as a secondary lift, and do **not** let
either regress recall in pursuit of a precision the footage can't support.

---

## 5. Hard constraints — respect these on any change here

1. **Recall-safety is the governing rule.** Every gate only drops the
   *confidently* bad (confirmed opponent, ball-shaped box). Anything unmeasurable
   (`unknown` colour, no appearance) survives to review. Never trade a real touch
   for a cleaner-looking review queue.
2. **Resolution ceiling is real.** Most footage is 640×360. Validate that a new
   signal actually fires at that scale before depending on it. The warning banner
   exists to tell the user, not to be engineered around.
3. **Compute is sparse by design.** Heavy work runs only at contact moments. A
   per-frame ReID pass across 90 minutes is off the table; per-contact (a few
   hundred) is fine.
4. **Grass masking is load-bearing.** Any colour read on a torso must go through
   the grass mask (`color.py`), or distant players read green.

---

## 6. Verification norms (how this project validates)

Follow the established pattern — **do not** claim a fix works from unit tests
alone:

1. **Unit tests** for the logic (`polyfut_v2/tests/`, `polyfut_video/tests/`).
   Run the full suite after changes to Stages 5–8; it must stay green.
2. **Ground it in real footage.** Completed runs leave a full audit trail at
   `%APPDATA%\PolyFut\exports\<job_id>\job_state.json` (the *packaged* app's data
   dir, outside the repo) — every touch's `player_bbox`, `color_dist`,
   `is_other_team`, `confidence`, etc. Re-score real moments against the actual
   video (`%APPDATA%\PolyFut\uploads\<token>.mp4`) before/after a change and show
   the numbers. This is how every prior fix here was verified.
3. **Front-end changes**: load in the browser preview, check the console for
   errors, verify the observable behaviour — the state lives in module-scoped
   `let` vars that can't be injected from `window`, so exercise via the real flow
   or read source, not by mutating globals.

---

## 7. Key files

| File | Role |
|---|---|
| `polyfut_v2/ball_model.py` | Soccer model + class map `{0:ball, 1:goalkeeper, 2:player, 3:referee}` |
| `polyfut_v2/pipeline/player_detector.py` | Detection; **only requests class `player`** today |
| `polyfut_v2/pipeline/player_contacts.py` | `nearest_player` (who touched) + `enrich_contact` (team gate) |
| `polyfut_v2/pipeline/color.py` | Grass-masked jersey HSV, `hsv_distance`, `hex_to_hsv`, `grass_fraction` |
| `polyfut_v2/pipeline/appearance.py` | `HistogramAppearance` (swap point for a ReID embedding) |
| `polyfut_v2/pipeline/scoring.py` | Orbital prior + tracklets + `transform` hook (motion continuity lives here) |
| `polyfut_video/pipeline/team_preview.py` | Per-match 2-team k-means (the "pick your team" clustering) |
| `polyfut_v2/pipeline/grouping.py` | Review-time team/look-alike grouping (not a detection gate) |
| `polyfut_v2/config.py` | All thresholds cited above |
| `docs/pipeline-v2.md` | Overall v2 design + rationale |
