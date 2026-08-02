# Detection & Identification — Issues and Solutions

**Status:** Issues 1–10 implemented; 15 proposed · **Scope:** Stages 5–8 of the v2 pipeline (and the
continuous ball path that feeds them) · **Related:** [pipeline-v2.md](pipeline-v2.md),
[`CLAUDE.md`](../CLAUDE.md)

This document turns the detection-capability assessment into concrete work items.
Each issue states the **problem**, the **limitation** that makes a naive fix fail
or insufficient, and the **solution that will be applied**. Solutions respect
recall-safety: only confidently-bad detections are dropped; ambiguity still
reaches human review.

---

## Issue 1 — Goalkeepers invisible, referees leak in as players

### Problem
The soccer model exposes classes `{0: ball, 1: goalkeeper, 2: player, 3: referee}`,
but [`player_detector.py`](../polyfut_v2/pipeline/player_detector.py) requests only
`classes=[player_class_id]` (class 2). Goalkeeper touches never appear. Referees
misclassified as `player` cannot be excluded by class and show up as touch
candidates on the review screen.

### Limitation
Class labels only help when the model actually fires that class at ~640×360. If
referee recall on class 3 is poor on real footage, class-based exclusion alone will
not clear all refs — colour/position heuristics remain a fallback, not a
replacement for requesting the class.

### Solution
1. Request classes `[goalkeeper, player, referee]` (1, 2, 3) from the player
   detector.
2. Treat `goalkeeper` as a valid contact candidate (keepers touch the ball).
3. Treat `referee` as never eligible for contact attribution — drop before
   `nearest_player` / enrichment.
4. Keep the existing ball-shape reject for class-`player` boxes that are really
   the ball.
5. Verify on real exports (`job_state.json` + source video) that class 3 fires on
   refs at 640×360; if recall is poor, keep colour/position soft signals as a
   secondary filter rather than pretending the class alone is sufficient.

**Primary files:** `player_detector.py`, `player_contacts.py`, `config.py`,
`tests/test_player_contacts.py`.

---

## Issue 2 — Wrong-team touches survive as “undecided”

### Problem
Team membership is a soft HSV distance to the user’s seed kit
(`≤60` mine, `>115` other, 60–115 undecided kept). Opponent kit from the team
picker is only used as a narrow tie-break. Wrong-team players in the undecided
band, and touches without a strong opponent centroid, still reach review.

### Limitation
A detection-model class named “other team” cannot work — kit colours change every
match. Team membership is a **per-match clustering** problem. Aggressively
dropping the undecided band would trade real touches for a cleaner queue and
violates recall-safety.

### Solution
1. Promote the existing per-match two-kit k-means
   ([`team_preview._kmeans_two_kits`](../polyfut_video/pipeline/team_preview.py))
   into a pipeline-wide **team model**: two HSV centroids computed once per match
   and reused at contact time (user’s picked centroid = my team; the other =
   opponent).
2. Classify every contact player into `team_a` / `team_b` / `official` /
   `unknown` inside `enrich_contact`:
   - nearest centroid within a confidence margin → `team_a` / `team_b`;
   - model class `referee` → `official` (from Issue 1);
   - unreadable colour → `unknown` (keep).
3. Gate in the orchestrator: drop only **confirmed opponents** and **officials**;
   always keep `unknown`.
4. Align review grouping thresholds with the detection gate so UI flags and
   pipeline drops do not disagree (`grouping_other_team_dist` vs
   `contact_other_team_dist`).

**Primary files:** `player_contacts.py`, `app_service.py`, `grouping.py`,
`config.py`, `team_preview.py` (reuse clustering, do not reinvent).

---

## Issue 3 — Nearest box wins even when it is not contesting the ball

### Problem
`nearest_player` picks the closest detection within `contact_max_player_dist_px=80`
(after the ball-shape reject). Sideline, bench, and near-graphic objects that
happen to be nearest the ball become touch candidates. Overlapping boxes that
contain the ball point all score distance zero; iteration order wins.

### Limitation
At 15px player height there is no reliable pose/foot model. A hard “must overlap
the ball” rule would drop real touches where the box is offset. Any new gate must
only reject when the contact is **confidently non-contesting**.

### Solution
1. Keep distance as the primary ranking signal.
2. Add a soft contesting check before accepting the winner:
   - prefer boxes whose lower third (feet/legs region) is nearer the ball than
     the box centre alone;
   - when multiple boxes contain the ball point, prefer the smaller / higher-
     confidence player box over a large sideline blob;
   - reject candidates whose box is almost entirely outside the ball ROI and
     whose colour is unreadable **and** far from both team centroids only when
     that combination is confident — otherwise keep for review.
3. Strengthen ball-vs-player confusion: if a “player” box heavily overlaps the
   current ball detection bbox and still looks ball-like on size/aspect, reject
   it even when only one of the current `_looks_like_ball` conditions holds
   (overlap with the ball detector is the extra evidence).
4. Do not introduce a min-area floor that blinds distant real players.

**Primary files:** `player_contacts.py`, `tests/test_player_contacts.py`.

---

## Issue 4 — Same-kit identity is colour-led and snaps across gaps

### Problem
You vs a same-kit teammate is scored as
`confidence = appearance × orbital`, where appearance is an H–S histogram
([`HistogramAppearance`](../polyfut_v2/pipeline/appearance.py)) and the orbital
prior is a soft floor (`orbital_floor=0.5`). Tracklets hard-cut at
`tracklet_max_gap_sec=3`. Identity is re-decided per contact instead of carried
between confident sightings. Appearance leads; motion only nudges.

### Limitation
Colour cannot separate same-kit teammates by construction. A richer hand-crafted
colour body descriptor was already tried and failed (worse separation than torso
HSV) — that path will not be repeated. At ~15px, same-kit ID is near the physical
ceiling; human review remains the precision layer. Motion continuity drifts over
long gaps, occlusions, and cuts, so it cannot be the only signal.

### Solution
1. **Primary — motion continuity as the backbone** (build first):
   - Between contact moments, track the anchored player with the same pattern as
     seed-clip tracking in [`seed_clips.py`](../polyfut_v2/seed_clips.py)
     (colour-locked nearest-centroid + merge/rejoin), so identity survives gaps
     that the 3s tracklet cutoff currently breaks.
   - Let a strong motion chain **override** a weak/ambiguous appearance score
     (invert today’s appearance-led product for well-anchored chains), while
     still flooring so a bad prior cannot zero out a real touch.
   - Past `orbital_max_gap_sec`, degrade to neutral as today.
2. **Secondary — appearance upgrade** (after motion work):
   - Swap `HistogramAppearance` for a learned torso ReID embedding behind the
     existing `AppearanceModel` protocol (OSNet-style or equivalent).
   - Run ReID only at contact moments (sparse), never per-frame across the match.
3. Keep review grouping as the UX propagation layer; do not pretend auto-ID
   replaces “me / not me.”

**Primary files:** `scoring.py`, `seed_clips.py` (reuse tracking pattern),
`appearance.py`, `config.py`.

---

## Issue 5 — Orbital reasoning breaks during camera pans

### Problem
[`scoring.py`](../polyfut_v2/pipeline/scoring.py) has a `transform` hook for
pitch/camera compensation, but it defaults to identity. Orbital distance is
computed in raw pixel space. During pans and zooms the motion prior is wrong,
so the strongest available same-kit signal is effectively disabled on much of
the footage.

### Limitation
Full pitch homography every frame is expensive and brittle on amateur / zoomed
broadcasts. A perfect world-frame transform is not required — a lightweight
inter-frame compensation that stabilises the ball neighbourhood is enough to
make orbital distances meaningful again.

### Solution
1. Implement a non-identity `transform` used by `score_contacts`: estimate a
   local affine / translation between consecutive analysed frames in the ball
   ROI (optical flow or feature match on the grass/pitch patch), and map
   contact positions into a pan-compensated space before orbital distance.
2. Reset the transform across shot boundaries (same rule as ball smoother shot
   resets).
3. If compensation confidence is low, fall back to identity for that step
   (recall-safe: do not invent motion).
4. Reuse patterns from [`review_track.py`](../polyfut_v2/review_track.py)
   (Lucas–Kanade) where applicable rather than adding a second tracking stack.

**Primary files:** `scoring.py`, `review_track.py` / small new helper under
`pipeline/`, `config.py`, `tests/` for orbital behaviour with a synthetic pan.

---

## Issue 6 — Ball path loses soft touches and fabricates stale positions

### Problem
Ball smoothing holds the last trusted box for up to `ball_hold_frames` instead of
interpolating. Stage 4 correctly ignores held samples for velocity, but long
holds still blur timing and hide inflections. Soft touches with little speed or
direction change never become candidates. High-confidence distant false positives
can still teleport unless they exceed the hard jump cap.

### Limitation
Ball recall on 640×360 is the upstream reliability ceiling — no contact logic
can recover a touch the ball detector never saw. Soft possession without a
kinematic signature is inherently invisible to a ball-anchored design; that is
accepted product scope, not a bug to “threshold away.”

### Solution
1. Replace pure hold with short-gap **linear interpolation** between trusted
   detections when the gap is within the current hold budget; keep hold-only
   behaviour when there is no forward anchor yet.
2. Continue excluding interpolated/held samples from Stage 4 velocity nodes
   (detected-only kinematics stay).
3. Tighten teleport rejection: require either spatial continuity *or* a second
   confirming detection before accepting a far high-conf jump.
4. Leave soft no-kinematic touches out of auto-candidates; do not invent contacts
   from player proximity alone (that would destroy the ball-anchored efficiency
   model). Surface low ball-detected-ratio via the existing warning path.

**Primary files:** `ball_smooth.py`, `ball_tracker.py`, `contacts.py`,
`tests/test_ball_smooth.py`.

---

## Issue 7 — Runtime cost is dominated by repeated decode + continuous ball inference

### Problem
A full match still spends most of its time on continuous ball inference (stride 4)
and on multiple independent video passes (shot filter, deadtime, ball track, seed
search, sparse contacts, on-demand review). Seed clip generation can add dozens of
player inferences per slot. Stage 5–8 lack separate timing, so regressions are hard
to see.

### Limitation
Dense per-frame player detection is deliberately off the table — bringing it back
would erase the v2 design win. Batching and decode reuse must not force large
frame buffers that blow memory on 90-minute files.

### Solution
1. Add per-stage timers in the orchestrator / `app_service` for: ball track,
   seed generation, sparse player+enrich, scoring, grouping — report them in
   job logs / `job_state.json`.
2. Where candidate contact frames are temporally close, batch player ROI crops
   into a single model call when the provider already has those frames decoded
   (preserve time order; cap batch size).
3. Avoid re-deriving appearance descriptors in grouping when Stage 7 already
   computed them — cache descriptor vectors on the contact / montage item.
4. Remove stray debug JSONL I/O from `team_preview` if still present.
5. Do **not** enable `team_filter_enabled`’s 3× detection window as the default
   path; Issue 2’s two-centroid gate should work on the single contact frame
   first.

**Primary files:** `orchestrator.py`, `app_service.py`, `player_detector.py`,
`grouping.py`, `team_preview.py`.

---

## Issue 8 — Corner kicks: everyone is on top of the ball, so attribution is a coin flip

### Problem
At a corner (and in any goalmouth scramble) six to ten players stand inside a few
metres of the ball. `nearest_player` still picks exactly one box, and the jersey
colour is then read off *that* box. Both decisions are close to arbitrary: the
"nearest" body is often a blocker rather than the toucher, and the kit sampled may
belong to the other team. Downstream this is silent — a wrong-team colour read gets
the contact **dropped** by the team gate, and a weak appearance score gets it
**auto-hidden**. The user never sees the clip, so a real touch simply disappears at
exactly the moment they most want to check it.

### Limitation
There is no way to fix the attribution itself: at 640×360 with overlapping bodies,
no colour or appearance signal separates who actually made contact. Nor can crowded
touches simply be force-accepted — half of them genuinely belong to someone else.
The only correct answer is to stop deciding and ask.

### Solution
1. **Measure the crowd.** During enrichment, count *eligible* players (same class
   and ball-shape filters as attribution) within `crowd_radius_px` of the ball, and
   flag the contact `crowded` past `crowd_min_players`. Take the max across the
   sampled window — a pack on any frame means the attribution is untrustworthy.
2. **Never drop a crowded touch on colour.** Both the always-on opponent gate in
   the orchestrator and `filter_my_team` exempt crowded contacts: the kit we read
   belongs to whichever body won the nearest-player race, which in a pack is often
   not the toucher. Officials still drop.
3. **Always route them to the human.** `build_montage` forces `review` status on a
   crowded contact regardless of confidence, and keeps the automatic verdict as the
   *pre-filled* decision — so an unreviewed crowded touch still resolves the way
   confidence said, and the user gets the chance to overrule rather than the burden
   of having to. Crowded touches draw on their own `crowd_max_review` budget so a
   long ordinary queue can never bury them.
4. **No bulk auto-settling.** The review UI's kit/look-alike propagation skips
   crowded clips in both directions — its groups come from the same untrusted
   attribution. Each crowded touch is judged individually, and is badged as such.

Config: `crowd_detect_enabled`, `crowd_radius_px=90`, `crowd_min_players=4`,
`crowd_force_review`, `crowd_keep_other_team`, `crowd_max_review=40`.

**Primary files:** `player_contacts.py`, `montage.py`, `orchestrator.py`,
`config.py`, `script.js`.

---

## Issue 9 — Referees and sideline coaches still get tagged as players

### Problem
Even after requesting the model's `referee` class (Issue 1), review still shows
tags on referees and coaches. Two gaps remain: (1) the model often mislabels a
ref as `player`, so class exclusion never fires; (2) coaches / bench staff have
no model class at all — they come through as `player` and win `nearest_player`
when the ball is near the touchline.

### Limitation
There is no reliable "coach" class to request. Class-3 referee recall at 640×360
is incomplete. Any colour heuristic for black/neon kits must not drop a real
player whose team actually wears black.

### Solution
1. **Feet-on-pitch gate.** Sample grass under each detection's feet. Bodies whose
   feet sit on dirt/track/concrete (and whose frame is otherwise a pitch scene)
   are labelled `sideline` and withheld from attribution — so a coach can't beat
   an on-pitch player, and a coach-only contact is dropped. Unmeasurable / non-
   pitch scenes never reject (recall-safe).
2. **Soft official-kit fallback.** Black or neon-yellow kits that match *neither*
   team centroid are labelled `official` even when `class_id` says player —
   catching refs the model missed. Never overrides a clear your-team match.
3. Crowded-contact exemptions still do **not** keep officials or sideline staff.

Config: `sideline_reject_enabled`, `sideline_min_feet_grass=0.18`,
`official_kit_reject_enabled`.

**Primary files:** `color.py`, `player_contacts.py`, `orchestrator.py`, `config.py`.

---

## Issue 10 — Match-long identity after seed; no false auto-hotspots

### Problem
After the user IDs themselves via seed clips, contacts with high appearance but
no motion link to that person still auto-accept into hotspots. Several wrong
`me` times near each other merge (`hotspot_gap_merge_sec`) into one long zone.
Review also kept boxing a body when tracking was lost or uncertain.

### Limitation
Dense per-frame tracking for 90 minutes is off the table (compute). Same-kit
appearance alone cannot prove "you". Identity must be carried sparsely at
contact moments via the orbital / seed sightings.

### Solution
1. Store seed **sightings** `(x, y, t)` from the 4 taps and bootstrap Stage 7's
   orbital from them so the chain starts from who the user clicked.
2. Mark contacts `identity_linked` only when motion-carried or continuity-
   confirmed inside the orbital / seed bootstrap. Cold-start strong appearance
   alone is not linked.
3. Auto-accept into hotspots requires `identity_linked` (`autoaccept_require_identity`).
   Unlinked high-conf contacts stay in review for an explicit "That's my touch".
4. Review UI: box only when `identity_linked`; hide when the short-clip LK track
   is lost; never invent a body box on the ball.

**Primary files:** `seed.py`, `scoring.py`, `montage.py`, `review_track.py`,
`script.js`, `config.py`.

---

## Issue 11 — A narrow grass hue band silently gated the whole pipeline

### Problem
On the exemplar TAS/ISM clip the run produced **one** hotspot, and it was a
phantom: the ball trajectory had locked onto a white shirt 250px from the real
ball. Root cause was not identification. `color.py`'s grass band started at hue
**32**, but that sunlit pitch has a modal hue of **28** — so only 11.5% of pitch
pixels registered as grass, and every gate built on the band misfired at once:

| Gate | Effect on that clip |
|---|---|
| `is_bbox_on_pitch_by_grass` (ball) | rejected **48.9%** of on-pitch ball positions |
| `is_off_pitch` (players) | discarded **84%** of on-pitch players as “sideline” |
| `jersey_hsv_from_crop` | masked **0%** of grass → torso colour is grass |

Measured across all 9 distinct test uploads, pitch hue is 28, 28, 35, 35, 35, 41,
41, 42, 42 — the band is fine on seven and catastrophic on the two shot at the
bright venue.

### Limitation
Lowering the constant is **not** the fix: aggregate gate-rejection keeps
improving monotonically down to hue 20, so the metric only measures how disabled
the gate is and cannot identify a correct floor. Two clips are also
floor-insensitive, so hue is not the only limiter. `team_classify.py:36` holds a
second, independent copy of the same constant, used by the team picker.

### Solution
1. **Ball gate inverted (done).** `is_bbox_on_foreign_surface` replaces
   `is_bbox_on_pitch_by_grass`: reject only on positive evidence of a non-pitch
   surface (saturated blue/red running track), never for lack of grass. Only
   saturated pixels vote; the ball's own box is excluded; an unreadable or mixed
   probe keeps the detection. No dependence on this pitch's turf hue.
   Config: `ball_foreign_surface_frac`, `ball_surface_min_colored_frac`,
   `ball_surface_check_half_px`.
2. **Feet gate must admit ignorance.** `feet_grass_fraction` returns a confident
   `0.0` from ~40px of boots and shadow; it should return `None`. `region.size
   < 16` is far too permissive to deliver the “unmeasurable → keep” its docstring
   promises. Consider disabling `sideline_reject_enabled` below ~540p.
3. **Derive the band per match**, from the modal hue of the dominant mid-frame
   region across frames the shot filter already samples (mode ± ~10). A prototype
   recovered the correct mode on 8 of 9 uploads with no per-venue tuning. Fix
   both copies of the constant.
4. **Let Stage 4's own flags matter.** `ContactCandidate.strength` and `from_gap`
   are computed and never read anywhere else, so the phantom (`from_gap=True`,
   `strength=0.45`) still reached `confidence=0.87` and auto-accepted.

**Not the fix:** shape-gating ball detections. Measured over 68 real detections
across 4 clips, aspect is 1.02–1.78 and max dimension 4–13px (median 5px), with
circularity tracking model confidence. At 5 pixels a white patch on a shirt *is*
a small round blob — there is no shape signal left to separate them. Trajectory
plausibility (reversal geometry) is the signal that distinguishes them; see
Issue 13.

Config: `ball_pitch_gate_enabled` (semantics changed), `sideline_min_feet_grass`.

**Primary files:** `color.py`, `ball_detector.py`, `player_contacts.py`,
`team_classify.py`, `config.py`.

---

## Issue 13 — The ROI search sustains its own false lock

### Problem
Inverting the ball pitch gate (Issue 11) raised raw detections on the exemplar
from 6.3% of analysed frames to 40.6%, and on a second clip from the same venue
to 98%. That number is **not recall**. Measuring the turn angle between
successive detections over 3117 links shows two populations:

| turn between steps | share of links | median speed |
|---|---|---|
| ≤ 30° (smooth — real ball) | 22.6% | 64 px/s |
| 170–181° (near-total reversal) | 45.0% | 456 px/s |

The top offenders are repeated ~117px steps at 0.133s intervals reversing ~175°,
recurring at many different timestamps: the tracker shuttling between two fixed
points. `roi_half_px=120` re-finds "the ball" within a fixed radius of the last
position every frame, so once it locks onto a false positive it keeps finding
something nearby and the lock feeds itself. The old grass gate had been
accidentally suppressing some of this by rejecting ~half of all positions.

### Limitation
`ball_confirm_jumps` was meant to catch exactly this and **never fires on this
footage**: 1-frame displacements top out at ~112px (p99), never reaching the
120px `ball_suspect_jump_px` trigger — the excursions sit just under it. Lowering
that constant is a per-clip tuning exercise of the kind Issue 11 warns about.
Also, a *single* out-and-back cannot be distinguished from a real return pass:
the ball reverses and travels back through where it already was in both cases.

### Solution
**Stage 3f, `ball_sanity.reject_pingpong` (done).** A post-pass over the finished
trajectory — it must see the sample *after* the suspect one, which is why it
cannot live in the online smoother. A link is suspicious when the step exceeds
`ball_pingpong_min_step_px`, reverses by ≥ `ball_pingpong_min_turn_deg`, and
*returns* (`|p3-p1| ≤ ball_pingpong_return_frac · |p2-p1|`). Only a run of
`ball_pingpong_min_alternations` consecutive suspicious links is acted on, so one
ambiguous out-and-back is left alone.

Every sample spanned by a confirmed run is demoted to a position-less miss, not
just one side: in an alternation there is no geometric way to say which endpoint
is the ball, and a velocity across the run is meaningless either way. Blanking
the span makes the pipeline say "position unknown here" rather than handing
Stage 4 an inflection to convert into a false touch — which is how the exemplar's
single hotspot was fabricated. `ball_sanity` counts are written to
`job_state.json` so a sparse run is explainable after the fact.

### Measured result (both clips from the bright venue)

| | exemplar | 2nd clip, same venue |
|---|---|---|
| detections before → after 3f | 913 → 900 (**13 blanked**) | 2206 → 870 (**1336 blanked**) |
| median turn before → after | 20.7° → 19.8° | **179.8° → 104.8°** |
| links reversing ≥170° | 6.3% → 5.6% | 61.2% → **25.1%** |
| Stage 4 candidates | 181 → 179 | 132 → 100 |

Read this honestly:

* **The pitch-gate inversion (Issue 11) is what helped the exemplar** — 142 → 913
  detections and 25 → 179 candidates, with a median turn of 20.7° and 57.8% of
  links turning ≤30°, i.e. genuinely ball-like motion.
* **Stage 3f is a no-op on the exemplar** (13 rejections) and does its work on the
  second clip, whose median turn of 179.8° means the path alternated almost every
  sample. It is justified by that clip, not by the exemplar.
* **That clip is still not trustworthy after the pass** — 25% of links reverse
  ≥170° and the median turn is 104.8°. 3f removes the worst of a false lock; it
  does not make a contaminated trajectory good.
* **The exemplar's phantom is NOT fixed.** Candidates still appear at 167.2s and
  167.73s (originally 168.0s). Its false lock was never an alternation — it drifts
  — so 3f has nothing to catch. Closing that failure needs Issue 11 item 4:
  **72 of the 179 exemplar candidates are `from_gap`** and nothing reads the flag.

Still open: `ball_conf_min = 0.07` admits a lot of the noise this pass then has
to clean up, and ~40% of detections sit below conf 0.15. Worth revisiting once
the trajectory is trustworthy — but measured across clips, not tuned on one.

Config: `ball_pingpong_reject_enabled`, `ball_pingpong_min_step_px`,
`ball_pingpong_min_turn_deg`, `ball_pingpong_return_frac`,
`ball_pingpong_max_gap_sec`, `ball_pingpong_min_alternations`.

**Primary files:** `ball_sanity.py`, `main.py`, `config.py`, `app_service.py`,
`orchestrator.py`.

---

## Issue 12 — Kits with more than one colour averaged to a colour they don't contain

### Problem
Team profiles carried a single hex, produced by medianing torso colours. A kit
that isn't flat — red/blue halves, hoops, a contrast sleeve — averages to purple,
a colour no player wears, so neither half matched it and real touches were
labelled other-team. The seed built from the user's own taps had the same bug via
`median_hsv`.

### Limitation
Colour still cannot separate same-kit players (§4.4), and it cannot separate two
teams that share a colour. On low-res footage the *swatch colours themselves* are
grass-contaminated (Issue 11), so multi-colour derivation gives several muddy
swatches rather than several correct ones until that lands.

### Solution
A kit is a **set** of colours, matched on its nearest member:
1. `color.cluster_hsv` keeps a kit's distinct colour modes instead of medianing
   them; `hsv_distance_multi` measures to the nearest colour; `kits_separable`
   decides two kits are distinguishable only when their *closest* cross pair is
   far apart, so a colour both teams wear can never drive an opponent drop.
2. `TargetSeed.kit_hsv_alts` / `opponent_kit_hsv_alts` with `my_kits()` /
   `opponent_kits()`. `kit_hsv` stays the dominant colour, so single-colour call
   sites, serialized sessions and hand-built test seeds are unchanged.
3. `classify_team`, `looks_like_official_kit` and `scoring.kit_compatible` all
   take colour sets — the last matters because a two-colour player reads as one
   colour on one touch and the other on the next, which broke the chain.
4. `team_preview._crops_to_hexes` sub-clusters each team's crops into up to 3
   colours, dominant first, dropping any colour claimed by both teams. The
   response keeps `hex` and adds `hexes`.
5. Team screen shows one chip per colour; each chip *is* a colour input, with a
   remove badge and an add button (max 3). Edited colours are sent as
   `my_team_hexes` / `opponent_hexes` and **added** to the tap-derived kit rather
   than replacing it — recall-safe, since extra colours can only ever match more.

Config: none added (clustering thresholds are local constants).

**Primary files:** `color.py`, `seed.py`, `player_contacts.py`, `scoring.py`,
`team_preview.py`, `app_service.py`, `server.py`, `script.js`, `style.css`.

### Follow-up: the split has to happen *inside* a crop, not just across them

Clustering across samples cannot recover a kit whose colours sit in the *same*
crop. `np.median` runs per channel, so a half-blue half-yellow torso medians to
hue 74 (green) before clustering ever sees it — eight identical two-colour crops
then produce one green mode. Two further problems surfaced on the way:

* **Hue is circular.** A red kit's pixels straddle the 0/180 seam, so the plain
  median put it at hue 90 — cyan — and a plain standard deviation read 88, i.e.
  maximally multi-coloured while being one colour. Hue is now combined with a
  circular *median* (rotate to the circular mean, unwrap, median, rotate back):
  correct across the seam **and** outlier-robust, which a circular mean is not.
* **A rejected split must not fall back to the blend.** A red/blue crop sits
  below the split bar, and returning the overall median there reported magenta —
  the original bug again. There are now two bars: above `_MODE_TRUST_HUE_STD`
  (20) the single answer becomes the *dominant mode*; above `_SPLIT_HUE_STD`
  (45), and only if the modes look like real dye, both colours are reported.

Calibrated on **924 real torso crops from 7 clips**, where single-colour kits
have hue spreads of p50=6, p90=20, p95=30, p99=42. Spread alone cannot decide —
a genuine red/blue kit only reaches ~34, inside that noise — so three gates do
the real work: both modes saturated (S≥70), both bright enough for hue to mean
anything (V≥70, since a near-black pixel computes a high S with a junk hue), and
a genuine hue separation (≥25 units) rather than a brightness difference. At
threshold 18 that measurement showed **24 false splits, every one a pair of dark
muddy colours** (`#2f3249` vs `#412c31`) from shadow against grass bleed; with
the gates and threshold 45 it is **0 of 924**, while a blue/yellow shirt (spread
70) still splits.

The honest cost: only strongly-opposed pairs split automatically. A red/blue kit
(~120° apart) falls back to its dominant colour, which is exactly today's
behaviour, and the second colour can be added by hand on the team screen.

The team picker needed the same treatment for a different reason: it now pools
every crop's non-grass pixels and clusters those, but Euclidean BGR distance
separates a sunlit shirt from a shaded one — unguarded it returned
`['#4c3d27', '#9f8650']`, one olive kit at two exposures. The same hue /
saturation / value gates are applied there (kept local: `polyfut_video` must not
depend on `polyfut_v2`).

Scope decision: **reference reads split, contact reads do not.** The seed and the
picker have many crops behind them, so a second colour can be confirmed; a
~6x12px contact crop would invent one. Contacts return their dominant colour and
match against the kit's full set.

Known limit: an achromatic pair (red/**white**, black/white) cannot be split by
hue at all — white has none. It reads as the chromatic half, which is the useful
answer. A green kit is also masked away as grass, unchanged by this work.

---

## Issue 14 — "Who touched it" is decided in pixels, so it means different things at each end of the pitch

### Problem
The rule that picks the toucher is `contact_max_player_dist_px = 80`: whoever's
box is nearest the ball, within 80 pixels. The catch is that a pixel isn't a
fixed distance. Near the camera one pixel is about 11cm; at the far touchline
it's about 62cm. So the same 80-pixel rule means:

| where the action is | what "within 80px" actually allows |
|---|---|
| near touchline | **7.9 m** |
| midfield | 22 m |
| far touchline | **49.5 m** |

When play is at the far side, a player **50 metres away** can win "nearest
player" and get tagged with your touch. `crowd_radius_px = 90` has the same
problem: it counts everyone within 56m out there (so it thinks every far-side
moment is a scramble) and only 9m near the camera (so it misses real ones).

This shows up in real exports. On the exemplar job's 69 touches with usable
geometry, converting each ball→attributed-player distance into metres at its own
position in frame:

* median gap **3.3 m**
* **36%** of attributed touches have the player more than **5 m** from the ball
* **10%** more than 10 m, **3%** more than 20 m

A player 10m from the ball did not touch it. That's roughly one attributed touch
in ten being physically impossible.

It explains both of the symptoms that actually bother users. **Other players in
your hotspots**: a far-side body is "closer in pixels" than you are and steals
the attribution. **Your touches going missing**: same event — once your touch is
tagged to the wrong body, the colour gate or the review queue disposes of it.

### Limitation
Fixing this needs to know where the pitch is, which needs a per-venue
calibration, which needs a human to click some landmarks. That's a real product
cost and it doesn't always work:

* Of two venues tested, **one calibrated cleanly and one did not.** The failing
  one (CTFA) has worn markings and three goals visible from neighbouring
  pitches, so landmarks get misidentified. Its video is not worse quality —
  it's actually *less* compressed than the one that worked. Detection quality
  and calibration quality are separate things.
* **A calibration can look perfect and be completely wrong.** Low residuals,
  insensitivity to click noise, and a clean rank check all passed a CTFA
  calibration that was visibly nonsense. The *only* check that has ever caught a
  bad one is drawing the pitch back on the frame and seeing whether the lines
  land on the real painted lines.
* Because calibration is optional and can fail, there has to be a path for
  uncalibrated video. That's a second code path, and it is unavoidable.

### Solution
1. **Make the camera registration better first** (`camera_motion.py`). Per-step
   estimate goes from median translation to a **similarity** (translation +
   rotation + scale). Measured: similarity beats fitting a full perspective
   transform per step, which blows up when you chain it (canvas 8.7× too wide;
   drift 5.17px vs **0.72px**). Also mask screen-fixed graphics — a scoreboard
   doesn't move when the camera pans, so its features argue "no camera motion".
   Masking lifted tracked-point counts 400 → 463. **This helps every video,
   calibrated or not, and needs no UI.**
2. **Calibration screen.** User clicks 6+ pitch landmarks on one or two frames.
   Solve for a **camera** — position, height, pan, tilt, roll, focal length —
   not a free 8-number transform. A free transform can describe pitches folded
   through the camera or shaped like bow-ties, and with sloppy clicks it lands
   there while still reporting tiny errors. A camera can't express those shapes
   at all.
   * Fit each click in **its own frame's pixels**, never through the
     registration. Going through it corrupted the recovered camera height by
     **43%** and the focal length by 26%.
   * Draw the pitch live on the frame while clicking. This is the check that
     works; everything else can lie.
   * Show the implied camera in plain words ("13m high, 17m back from the
     touchline") so the user can sanity-check it against where they stood.
   * Warn when there are too few landmarks to prove anything (7 unknowns, so 5
     clicks leaves almost no slack and a good-looking error is meaningless).
3. **Convert the attribution gates to metres.** `contact_max_player_dist_px` and
   `crowd_radius_px` become metre thresholds applied after projecting the
   player's **feet** (box bottom-centre) onto the pitch. Feet, not the box
   centre or the ball — the maths only maps things standing on the ground.
   Recall-safe in the right direction: it *loosens* near the camera and
   *tightens* far away, so it removes impossible attributions rather than
   dropping close ones.
4. **Fall back silently.** No calibration, or a calibration that fails its
   quality check → behave exactly as today. No user-visible mode, no badge.
5. **Record what's needed to check this later.** `job_state.json` currently
   stores raw pixel positions while scoring runs on pan-compensated ones, and
   camera panning (~30px/s) is about the same size as the whole signal — so the
   motion prior cannot be evaluated at all from saved runs. Persist the
   compensated (and, when calibrated, the pitch) position. Free, changes no
   behaviour, and makes the next question answerable.

Only the homography **at contact moments** is needed for this, which fits the
existing sparse-compute budget. No continuous tracking, no per-frame passes.

### What this deliberately does not include
* **Metric orbital + velocity prediction.** Measured gain is real but smaller
  (ambiguity at a 2s gap: ~100% today → 49% with velocity prediction in
  pan-compensated pixels → 34% in metres) and it needs short bursts of player
  detection around each contact (~3–5× today's detections). Worth revisiting
  once item 5 makes it measurable.
* **Contacts from "player and ball overlap on the map."** Rejected on compute
  burden. Also sidesteps the fact that the map only works for things on the
  ground — a ball 2m in the air lands roughly 8m from where it really is.
* **Fitting the pitch dimensions.** Built and tested; it did *not* fix the
  failing venue (error dropped 25× while the overlay stayed just as wrong — a
  textbook case of extra parameters flattering a bad fit). Ship it off by
  default, as an opt-in for grounds known to be non-standard.

### Measured, for reference
* Good calibration accuracy: **~3.7px rms / ~1.7px median** ≈ 0.2–1.0m on the
  pitch, from 9 landmarks on one frame.
* One pixel is worth **5.7× more distance** at the far touchline than near the
  camera.
* Of player pairs 10–30px apart in the image, **51–66% are genuinely 3m+ apart**
  on the pitch. Those are the identity mix-ups this can fix. Pairs that really
  are within 2m are only **1.4%** of all pairs, and nothing fixes those.
* 99.3% of detected players project onto the pitch on a calibrated clip — a
  sanity check nothing forced.

### Do not repeat these
* **Tightening `orbital_base_px` / `orbital_growth_px_s`.** Checked against real
  human "me / not me" labels across 6 videos: displacement separates you from
  other players with **AUC 0.539** (0.5 = coin flip), and the current gate
  already drops **19% of real touches**. No setting helps. Note the measurement
  is confounded by item 5 above and should be redone once that lands.
* **Trusting a calibration because the numbers look good.** See Limitation.

Config: `contact_max_player_dist_m`, `crowd_radius_m`, plus a stored per-venue
calibration (camera pose + pitch size) and a quality threshold for the fallback.

**Primary files:** `camera_motion.py`, new `pitch_calibration.py`,
`player_contacts.py`, `app_service.py`, `script.js` (calibration screen),
`config.py`, `job_state` schema.

---

## Issue 15 — Where you are is only known at contact moments, so motion can't carry identity

### Problem
The pipeline knows where the ball is on every analysed frame and where *you* are
only at the ~180 moments the ball was contacted. Everything in between is blank.
That's what forces identity to be re-decided from appearance at each touch —
and appearance is the signal that cannot separate same-kit players at 26px.

It also means the orbital prior (Issue 4/5), which is the strongest identity
lever available, is guessing across gaps it doesn't have to guess across. The
config admits it: `tracklet_max_gap_sec = 3.0`, on the assumption that a player
cannot be followed for longer than that.

That assumption is wrong, and measurably so. Running the project's own
colour-locked tracker over a contiguous 60s window of `b48758eb195e`:

| | |
|---|---|
| tracks formed | 62 |
| players on screen (avg) | 18.6 |
| **mean uninterrupted follow** | **40.7 s** |
| median track lifetime | 48 s of the 60 s window |
| tracks surviving the whole window | 13 of 62 |
| tracks lasting > 30 s | 46 of 62 |

Forty seconds, not three. There is roughly **13× more continuity available than
the pipeline currently reaches for**, and it is being thrown away.

There's a second, unrelated saving sitting in the same place. The soccer model
emits `ball`, `goalkeeper`, `player` and `referee` from **one forward pass** —
the class list is a post-NMS filter, not a cheaper inference. But the ball path
runs an ROI pass and then re-scans the full frame whenever the ROI misses:

```
451 analysed frames -> 673 network calls  (1.49 per frame)
roi_hits 182 · roi_misses 249 · full_scans 242 · skipped_full_scans 27
```

So we pay for 1.49 passes per frame and extract *only the ball* from them, then
pay again for player detection at contacts. One full-frame pass per frame would
cost **1.00 call/frame** and yield the ball **and every player**.

### Limitation
The obvious conclusion — "so track the player instead of the ball" — does not
survive contact with the numbers.

* **Cheap dense tracking is not accurate enough.** Sparse optical flow (LK,
  already in the codebase for camera motion) costs **2.4–3.7 ms/frame** against
  **98.7 ms** for detection — about 30× cheaper, and dense 30fps flow for a
  90-minute match would be ~8 minutes of compute. But following six real players
  from `b48758eb195e`, **four were lost inside one second** (0.5s, 0.5s, 0.8s,
  1.0s; the best two lasted 4.0s and 16.5s), median IoU against re-detection
  0.27–0.80. A 21–28px player has too few stable corner features. The cost
  budget is fine; the tracker isn't.
* **Accurate dense tracking is detection, and detection is the expensive thing.**
  The 40.7s figure above comes from detect-every-frame plus association. At
  98.7 ms/frame that is 4.4 hours for a 90-minute match at 30fps, or 1.1 hours
  at today's 7.5fps cadence.
* **Continuity is not identity.** A track can survive 60s having silently
  swapped players at a crossing. We have no ground truth to measure how often
  that happens, and the colour lock meant to prevent it is **inert at the
  bleached venue**, where every kit reads as turf (Issue 11).
* **Hours-long identity is arithmetic, not effort.** At ~40s per unbroken track a
  90-minute match needs about **135 handoffs**. Identity survives only if every
  one is right: at 99% per handoff that's `0.99^135 ≈ 26%`; at 95%, `≈ 0.1%`.
  Public benchmarks agree this is unsolved — SoccerNet-Tracking evaluates on
  **30-second** sequences (best entry 75.6 HOTA) and keeps its 45-minute video
  as a separate long-term problem; the 2025 game-state challenge tops out near
  64 GS-HOTA, on broadcast footage rather than 640×360 at 440 kbps.
* **Dropping the ball ROI to fund this costs ball recall.** Measured on the same
  clip: `detected_ratio` **0.574 → 0.412**, a 28% loss. The ROI is not waste —
  it buys recall on a small object.

### Solution
Take the continuity, not the anchor. Four steps, smallest first; each is
independently useful and none bets the product on same-kit re-ID.

1. **Fuse the passes.** One full-frame inference per analysed frame, then read
   ball *and* players out of the same result. Costs 1.00 call/frame against
   today's 1.49 and makes player positions effectively free. Must be validated
   against the ROI recall loss above — the likely recovery is to keep an ROI
   pass but centre it on **the tracked player** rather than the last ball
   position, which is a better prior once you know where the player is.
2. **Associate detections into tracklets** (ByteTrack-style: IoU + motion,
   greedy, microseconds — the detection is already paid for). Produces the ~40s
   continuous "where you are" the measurements show is available.
3. ~~**Feed the tracklet into the orbital prior**~~ — **tried and refuted, see
   below.** Not implemented; `scoring.py` is untouched.
4. **Keep the ball as the anchor and keep human review for identity.** The
   tracklet informs the prior; it never becomes the source of truth.

### Step 3 was measured and does not work

The hypothesis was that the orbital prior grows its radius with the gap only
because it doesn't know where the player went, so a tracklet spanning that gap
would let it start from a fresh position instead. Measured on 120s of
`b48758eb195e` (901 samples, 157 tracks, 80 contact candidates, 79 gaps):

| carry tolerance | gaps bridged | tracked position closer than the stale anchor | **orbital prior improved** |
|---|---|---|---|
| 1.5 m | 23 (29%) | 12/23 (52%) | **0 / 23** |
| 4.0 m | 19 (24%) | 9/19 (47%) | **0 / 19** |
| 8.0 m | 5 (6%) | 1/5 (20%) | **0 / 5** |

Three reasons, all structural:

* **The prior is already saturated.** Median distance from the stale anchor to
  the next contact is 60.7px on 0.5–2s gaps, against a radius of
  `80 + 60·gap` ≥ 110px. It is already 1.0, and a better position estimate
  cannot improve on 1.0. It could only help by *lowering* the prior for the
  wrong body — i.e. discrimination, not prediction.
* **It bridges the wrong gaps.** 40% of sub-0.5s gaps (where the prior needs no
  help), 8% of 2–5s gaps and **0% above 5s** (where it is weakest). Exactly
  backwards from useful.
* **A looser tolerance makes it worse, not better.** `track_at` refuses when two
  tracks are equally close, and at a contact several players usually are — so
  widening the gate converts matches into ambiguity refusals.

This is consistent with the existing "do not repeat" note that displacement
separates you from other players with **AUC 0.539**. A better-measured
displacement is still displacement.

**What steps 1–2 leave behind:** harvested players (~1% cost, ball provably
unchanged) and tracklets (mean life 16.0s). Both work; neither has a consumer.
Keep them only if a *measured* use appears — do not wire them in on the
assumption that more information must help somewhere.

### Measured, for reference
* Player work is **3.4%** of a run (`player_enrich` 77.3s vs `ball_tracking`
  2071.3s). The pipeline does **not** track all players — that was the v1→v2
  change — so "stop tracking everyone" saves nothing that is currently spent.
* Full-frame detection **98.7 ms/frame**; LK optical flow **2.4–3.7 ms/frame**.
* Network calls today: **1.49 per analysed frame**; fused design: **1.00**.
* Track lifetime with detect+associate: mean **40.7s**, median **48s**, 13 of 62
  surviving a full 60s window.
* LK flow on 21–28px players: **4 of 6 lost within 1 second**.
* Dropping the ball ROI: `detected_ratio` **0.574 → 0.412**.

### What this deliberately does not include
* **Making the player the anchor.** One bad handoff out of ~135 silently
  reassigns the rest of the match. This is the failure already recorded on job
  `e7efd5ac4bc3`, where a seed on the wrong person made all 12 auto-accepted
  touches wrong and put 7 of 11 hotspots before the user came on — as a seeding
  accident. Anchoring on the player makes it the architecture.
* **A learned ReID embedding to bridge handoffs.** Still the right §4.2 upgrade
  eventually, but it cannot be evaluated honestly while kit colour reads as turf
  at the bleached venue (Issue 11) — there is no trustworthy label to score it
  against.
* **Per-frame (30fps) tracking.** 4.4 hours per match. The 7.5fps cadence
  already yields 40s tracks; buying 4× the frames to extend that is not
  supported by any measurement yet.

### Do not repeat these
* **Sparse optical flow as the between-detection tracker on this footage.**
  Measured: 4 of 6 players lost inside a second at 21–28px. The cost is
  attractive and the result is unusable; do not re-try it without first fixing
  resolution.
* **Feeding tracklets into the orbital prior.** Built and measured: it improved
  the prior on **0 of 23** bridged gaps at any tolerance, because the prior is
  already 1.0 on the gaps it manages to bridge and bridges 0% of the gaps over
  5s where it isn't. See "Step 3 was measured and does not work".
* **Camera compensation inside the association gate.** Measured: 109 tracks at a
  12.9s mean life with it against 89 at 16.0s without, because the camera moves
  a median 2.79px between harvested frames while the gate already allows ~39px —
  all noise, no information. Separate from the orbital's use of it over
  multi-second gaps, which is not in question.
* **Assuming the pipeline tracks all players.** It has not since v2. Any
  proposal justified by "stop tracking everyone" is aimed at 3.4% of runtime.
* **Trusting track lifetime as evidence of correct identity.** Lifetime is
  measurable here; correctness is not, and the two are not the same.

Config: `tracklet_max_gap_sec`, `continuity_max_gap_sec`, `orbital_*`, plus new
knobs for the fused pass and association gate.

**Primary files:** `ball_detector.py`, `player_detector.py`, `fast_infer.py`,
new tracklet/association module, `scoring.py`, `orchestrator.py`, `config.py`.

---

## Explicit non-goals (honest ceiling)

These will **not** be treated as solvable end-states in this work:

- Reliable automatic you-vs-same-kit-teammate ID on ~15px players without human
  review.
- Another hand-crafted multi-region colour descriptor for ReID (already failed).
- Retraining the detector to output “other team” as a class.
- Trading recall for a tidier review queue by hard-dropping undecided colour or
  weak appearance.
- **Unbroken single-player identity across a full match.** ~40s of continuous
  tracking is available and worth using (Issue 15); the ~135 handoffs a
  90-minute match needs are not survivable at any per-handoff accuracy we could
  honestly claim, and the public benchmarks evaluate on 30-second clips for the
  same reason.

---

## Implementation order

1. **Issue 1** — keeper/ref classes (smallest, highest accuracy ROI).
2. **Issue 2** — per-match team model gate.
3. **Issue 3** — contesting / ball-overlap attribution.
4. **Issue 5** — pan-compensated orbital `transform` (unlocks motion signal).
5. **Issue 4** — continuity tracking between contacts + optional ReID swap.
6. **Issue 6** — ball interpolate + teleport tighten.
7. **Issue 7** — timing, descriptor reuse, cautious batching.
8. **Issue 8** — crowded-contact flag → guaranteed human review.
9. **Issue 9** — sideline feet-on-pitch + soft official-kit reject.
10. **Issue 10** — seed-orbital identity link + gated auto-accept + hide box when lost.
11. **Issue 11** — grass-band calibration. Ball gate inverted (done); feet gate
    honesty, per-match band derivation, and `from_gap` reaching auto-accept still
    open. Do these *before* any further identification work — the numbers that
    work would be tuned against are produced by these gates.
12. **Issue 12** — multi-colour kits (done). Swatch *accuracy* still blocked on
    Issue 11.
13. **Issue 13** — ROI false-lock blanking (done). Revisit `ball_conf_min` after,
    and only with per-clip numbers.
14. **Issue 14** — pitch calibration so "who touched it" is judged in metres.
    Do it in this order: registration upgrade and the `job_state` position
    fields first (both are small, help every video, and need no UI), then the
    calibration screen, then flip the attribution gates to metres.
15. **Issue 15** — continuous player position feeding the orbital prior. Do the
    fused pass first (it is a speed win on its own and can be measured against
    ball recall in isolation), then association, then raise the tracklet gap and
    wire it into scoring. Step 4 is a decision to *not* build something, and
    costs nothing. Do not start this before Issue 11 lands: the colour lock that
    keeps association from swapping players is inert while kits read as turf.

Verification for every issue: unit tests stay green, then before/after numbers on
real `%APPDATA%\PolyFut\exports\<job_id>\job_state.json` moments against the
source upload — not unit tests alone.
