"""Central configuration for the v2 (ball-anchored) pipeline.

Only the knobs needed by the stages built so far are defined here; later steps
append their own sections. Stages 1-3 reuse the v1 shot-filter / deadtime code,
so ``v1_config`` projects the shared knobs onto a v1 ``PipelineConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineV2Config:
    # --- Stage 1: decode ---
    # The width everything is analysed at, and the unit for every ``_px``
    # threshold below. 640 is the reference the constants were tuned against.
    target_width: int = 640
    # Let a higher-resolution upload actually be analysed at higher resolution.
    #
    # Without this the width is pinned to 640 and every upload is downscaled to
    # it on decode, so uploading 720p or 1080p changes nothing but decode cost —
    # while the low-resolution warning tells the user that higher resolution
    # "gives dramatically better results". That advice was not deliverable by the
    # code that printed it.
    #
    # On 640-wide source this is a strict no-op (min(640, cap) == 640), which is
    # every clip currently on disk, so it cannot alter existing behaviour.
    analysis_follow_source: bool = True
    # Ceiling on that. Cost scales with area: 1280 is 4x the pixels of 640, and a
    # 5-minute clip already takes ~55 min on this CPU. See ``for_frame_width``.
    analysis_max_width: int = 1280
    # Ball is analysed on every sampled frame (Stage 3, the continuous cost — the
    # single biggest runtime term). 4 → ~7.5 analysed fps at 30 fps source: still
    # fine for Stage 4 kinematics, and keeps a full 94-min match to ~2.7h with the
    # OpenVINO soccer model on a CPU (vs ~18h at every_n=3 on PyTorch).
    ball_sample_every_n: int = 4
    # Coarse stride for the cheap shot-filter pass.
    shot_filter_sample_every_n: int = 10

    # --- Playing-time window ---
    # The user declares the ranges they were actually on the pitch; the whole
    # pipeline runs inside them. Each range is padded by this much on both
    # sides before bounding decode / clipping live shots, because the handles
    # are dragged by eye against a scrubbing preview and are only approximate —
    # recall-safety, same rule as every other gate here. Seed-clip moments are
    # NOT padded: a seed from a moment the user wasn't playing is precisely the
    # contamination the window exists to prevent.
    playing_time_pad_sec: float = 45.0

    # --- Stage 2: shot filter (heuristics, mirrors v1 defaults) ---
    cut_hist_threshold: float = 0.45
    green_ratio_min: float = 0.22
    motion_smooth_max: float = 2.5
    graphic_uniform_ratio: float = 0.35

    # --- Stage 3a: deadtime ---
    deadtime_motion_threshold: float = 1.2
    deadtime_min_duration_sec: float = 60.0

    # --- Stage 3b: ball detection (pluggable; default COCO YOLO) ---
    # Swap ``ball_weights`` for a soccer-specific ball model (and set
    # ``ball_class_id`` to that model's ball class) to drop it in.
    ball_weights: str = "yolov8s.pt"
    ball_class_id: int = 32  # COCO "sports ball"
    device: str = "cpu"
    conf_threshold: float = 0.25
    ball_conf_min: float = 0.07
    # Inference size for the small ROI crop vs. the full-frame re-acquire.
    ball_imgsz: int = 416
    ball_full_imgsz: int = 640

    # Call the compiled OpenVINO model directly instead of going through
    # Ultralytics' predictor. Measured on this project's soccer model: a full
    # predict() is 69.3ms of which only 39.6ms is the network — the rest is
    # letterbox rebuilding, a torch round-trip and Results construction that
    # nothing downstream reads. The bypass reproduces the same arithmetic in
    # numpy (verified to return identical detections) for ~8.6ms, and ball
    # tracking is ~91% of a run. Set False to go back through Ultralytics.
    fast_infer_enabled: bool = True
    # Keep the players out of the full-frame ball scan instead of discarding
    # them. The model emits ball + keeper + player + referee from ONE forward
    # pass (the class list is a post-NMS filter), and on a measured run the
    # full-frame re-acquire already fires on 242 of 451 analysed frames — so
    # roughly half the video's player positions are already paid for and thrown
    # away. Costs no extra inference and cannot change the ball result: the
    # confidence filter runs on the max class score before the class list is
    # applied, and NMS offsets boxes by class. Issue 15, step 1.
    harvest_players_from_ball_pass: bool = True

    # --- Stage 3b: ROI search around last known ball position ---
    # A region-of-interest around the last ball gives higher effective
    # resolution on a tiny object and is far cheaper than full-frame every step.
    roi_enabled: bool = True
    roi_half_px: float = 120.0  # half-width of the square ROI, in resized px
    # On an ROI miss, re-scan the full frame this same step (vs coasting on the
    # stale position). Essential for fast-ball / small-pitch / zoom-changing
    # footage; near-free when the ROI usually hits (slow-ball wide shots).
    roi_fallback_full: bool = True
    # Miss-storm backoff: after this many consecutive misses, the expensive
    # full-frame re-acquire runs only every ``ball_miss_backoff_stride``-th
    # call instead of every sampled frame. On low-recall footage (360p) miss
    # runs dominate runtime — this cuts most of that cost for at most a
    # (stride−1)-frame re-acquire delay. 0 disables the backoff.
    ball_miss_backoff_after: int = 4
    ball_miss_backoff_stride: int = 3

    # --- Stage 3c: temporal hold / interpolation across YOLO misses ---
    # Reuses polyfut_video.pipeline.ball_smooth (BallSmoother).
    ball_hold_frames: int = 10
    ball_max_jump_px: float = 500.0
    # Confidence-aware teleport gate (see BallSmoothConfig). Rejects a ball
    # detection that leaps far AND is low-confidence — the false-positive
    # pattern that fabricates phantom touches at spots the ball never reached.
    ball_suspect_jump_px: float = 120.0
    ball_suspect_jump_conf: float = 0.30
    # Two-frame confirmation for far jumps (Issue 6): a detection beyond
    # ``ball_suspect_jump_px`` is only committed once a SECOND detection agrees
    # with it — either landing near the same spot, or continuing along the same
    # heading at a comparable speed (a genuinely fast ball). A lone flicker on a
    # sock / line marking never gets that agreement, so it can no longer
    # fabricate a phantom touch, while a real clearance costs one sampled frame.
    ball_confirm_jumps: bool = True
    ball_confirm_dist_px: float = 120.0     # "landed near the same place"
    ball_confirm_angle_deg: float = 45.0    # "kept going the same way"
    ball_confirm_speed_ratio: float = 1.6   # and not implausibly faster
    # Replace frozen hold positions with linear interpolation between the two
    # bracketing real detections. Held samples stay flagged interpolated (Stage 4
    # still refuses them as velocity nodes); this only stops a 10-frame hold from
    # reporting one repeated position where the ball was demonstrably moving.
    # A trailing hold with no next detection keeps the frozen position.
    ball_interpolate_holds: bool = True

    # --- Stage 3f: physically-impossible excursion rejection ---
    # The ROI search re-finds "the ball" within roi_half_px of the last position
    # every frame, so once it locks onto a false positive (sock, line marking,
    # bright patch on a shirt) the lock sustains itself and the path becomes an
    # alternation between two points. Measured on two real 640x360 clips (3117
    # links) the turn angle is bimodal: 22.6% of steps turn <=30 deg at a median
    # 64 px/s (real ball), 45.0% turn 170-180 deg at a median 456 px/s (the lock).
    # A ball cannot reverse and return on consecutive 0.13s samples, so the
    # out-and-back population is dropped. Note ball_confirm_jumps below never
    # fires on this footage: 1-frame displacements top out at ~112px, under its
    # 120px trigger.
    ball_pingpong_reject_enabled: bool = True
    ball_pingpong_min_step_px: float = 40.0    # ignore small jitter
    ball_pingpong_min_turn_deg: float = 165.0  # near-total reversal
    ball_pingpong_return_frac: float = 0.35    # |p3-p1| <= this * |p2-p1| ⇒ came back
    ball_pingpong_max_gap_sec: float = 0.6     # don't link across a blind stretch
    # A SINGLE out-and-back is left alone: a real return pass reverses and travels
    # back through where the ball already was, and no three-point test separates
    # the two. Only a repeat — A→B→A→B on consecutive samples — is impossible, so
    # this many consecutive suspicious links are required before acting.
    ball_pingpong_min_alternations: int = 2

    # --- Stage 3d: camera-motion (pan) compensation for the Stage 7 orbital ---
    # Orbital distances are meaningless in raw pixels while the camera pans, so
    # the motion prior — the strongest same-kit identity signal — is effectively
    # off on panning footage. A cheap Lucas-Kanade median background shift per
    # analysed frame accumulates into a pan-compensated space. Never invents
    # motion: a low-confidence estimate contributes zero shift.
    camera_motion_enabled: bool = True
    camera_motion_every_n: int = 2       # of analysed frames (stride 4 → every 8 source)
    camera_motion_width: int = 320       # downscale for flow (cost, not accuracy, driver)
    camera_motion_min_points: int = 8    # fewer good points → not confident
    camera_motion_max_shift_px: float = 200.0  # implausible per-step shift → ignored
    camera_motion_exclude_half_px: float = 60.0  # mask around the ball (players/ball bias)

    # --- Stage 3e: ball pitch/side-line gate ---
    # Spare balls on the sideline can be correctly detected as the soccer ball,
    # but they are outside the pitch and still pollute the ball trajectory.
    # We reject those by requiring that a small region around the detected ball
    # center is positively a NON-pitch surface (running track, painted
    # concrete). Asking "is this grass?" instead required the turf to fall inside
    # a narrow hue band, and on bright sun-bleached pitches (hue ~28 vs a band
    # starting at 32) that rejected ~49% of real on-pitch ball positions. The
    # test now only fires on positive evidence of another surface, so it carries
    # no dependence on this pitch's exact turf colour.
    ball_pitch_gate_enabled: bool = True
    # Half-size of the square probe (in resized target_width px) around the ball.
    ball_surface_check_half_px: float = 18.0
    # Fraction of *saturated* probe pixels whose hue is outside any plausible
    # pitch hue before the detection is rejected. High: a mixed sample (ball on
    # a touchline, grass plus a red hoarding behind) must survive.
    ball_foreign_surface_frac: float = 0.70
    # The probe must be at least this saturated overall to get a vote at all.
    # Greys, white lines, shadow and the ball itself carry no hue information, so
    # a washed-out sample abstains (→ keep) rather than guessing.
    ball_surface_min_colored_frac: float = 0.25

    # --- Stage 4: contact candidates (kinematics on the trajectory) ---
    # Speeds are in px/s on the resized (target_width) frame; thresholds are
    # deliberately generous (high recall) and will be tuned on real trajectories
    # once a soccer ball model is plugged in.
    contact_min_speed_px_s: float = 30.0   # noise floor — ignore near-stationary jitter
    contact_dir_change_deg: float = 50.0   # heading swing that flags a deflection
    contact_stop_ratio: float = 0.4        # speed_after/speed_before below this = a trap
    contact_speed_spike_ratio: float = 2.0  # speed_after/speed_before above this = a kick
    contact_merge_sec: float = 0.3         # collapse events within this window into one
    contact_gap_sec: float = 0.5           # velocity interval longer than this = unreliable
    # Compute budget: a single player touches the ball ~30-80x/match, over-
    # generated to a few hundred, so 600 is already generous. If Stage 4 yields
    # more (noisy trajectory / demo ball), keep only the strongest — each
    # survivor costs a sparse player-detection pass (the dominant runtime term),
    # and no human reviews hundreds of clips anyway.
    max_candidates: int = 600
    # Duration-scaled cap (min with max_candidates, floored at 60): all players
    # combined touch the ball ~15-20×/min, so 40/min is >2× headroom for real
    # touches plus ambiguity. A noisy 3-min clip goes 600 → 120 candidates,
    # directly bounding Stage 5-6 runtime. 0 disables duration scaling.
    max_candidates_per_min: int = 40
    # Time-fair selection window: budget is spread round-robin across windows
    # of this size so a noisy phase can't consume every slot (see
    # contacts.cap_candidates).
    candidate_fair_window_sec: float = 30.0
    # Centered position smoothing (samples). Real detector output jitters, and a
    # synthetic sweep showed window=3 lifts noisy-trajectory recall ~0.56→0.74 at
    # equal precision (5 over-smooths). Provisional — retune on real footage.
    contact_smooth_window: int = 3

    # --- Stage 0: multi-sample seed ---
    seed_sample_count: int = 4   # UI target (up to 4, min 2; 1 warns → weak gallery)
    seed_min_samples: int = 2

    # --- Stage 5: sparse player detection (pluggable; default COCO person) ---
    player_weights: str = "yolov8s.pt"
    player_class_id: int = 0     # COCO "person"; soccer outfield player = 2
    # Soccer model extras (None → COCO / single-class mode). When set, the
    # detector requests keeper+player+ref; keepers are valid contact candidates,
    # referees are never attributed as the touching player.
    goalkeeper_class_id: int | None = None
    referee_class_id: int | None = None
    player_conf_min: float = 0.25
    player_imgsz: int = 640
    # NMS overlap threshold for player detection. Ultralytics defaults to 0.7,
    # which is permissive enough that ONE player often survives as two or three
    # overlapping boxes — which then become separate tracks and separate
    # tappable markers a few pixels apart. Measured on real frames: 0.7 gave 14
    # overlapping pairs across 9 frames, 0.45 gave 1 (76 -> 65 boxes).
    player_nms_iou: float = 0.45
    # ROI around the ball for the sparse player pass (0 → full frame).
    player_roi_half_px: float = 160.0
    # Max ball-to-player distance (px) to count a player as the one in contact.
    contact_max_player_dist_px: float = 80.0

    # --- Stage 5b: metric attribution (needs a pitch calibration) ---
    # A pixel is not a fixed distance on the ground: ~11cm near the camera,
    # ~62cm at the far touchline. So the 80px gate above allows 7.9m of pitch
    # near the camera and 49.5m far away, and a player 50m from the ball can win
    # "nearest player". Measured on a real export, 36% of attributed touches had
    # the player >5m from the ball and 10% >10m. When the user has calibrated the
    # pitch, judge that distance in metres instead. Issue 14.
    #
    # 8m is deliberately equal to what 80px already means NEAR the camera, where
    # the pixel gate is sane. So this changes nothing in the near field and only
    # tightens the far field: it removes impossible attributions rather than
    # dropping close ones.
    contact_max_player_dist_m: float = 8.0
    crowd_radius_m: float = 9.0        # 90px near the camera is ~8.9m
    # The pitch map is a GROUND-plane map, and the ball is often not on the
    # ground. An airborne ball projects further from the camera than it really
    # is, which can only inflate a distance — i.e. cause a rejection. A player
    # this close to the ball *in the image* is plausibly the toucher whatever the
    # metric says, so the metric gate never overrules image adjacency. Protects
    # headers and close control.
    contact_metric_override_px: float = 25.0
    # A calibration whose typical landmark error exceeds this is not trusted and
    # the pipeline silently falls back to the pixel gates. Reference: a good
    # calibration on real footage scored ~1.7px median, a visibly broken one 3px+
    # — but note that residuals alone cannot prove a calibration is right, so
    # this only screens out the obviously bad.
    calibration_max_median_px: float = 6.0
    # The soccer model sometimes classifies the ball itself as "player" — a
    # small, roughly-square box sitting almost exactly on the ball. Since a
    # real "nearest player" search is centered on the ball, such a box has
    # near-zero distance and would otherwise always win. Reject a "player" box
    # as ball-shaped only when it's BOTH short AND not clearly taller-than-wide
    # (real people, even small/distant ones, are reliably one or the other —
    # empirically calibrated against real broadcast footage: every observed
    # ball misclassification was under 13px tall with an aspect ratio under
    # 1.8, while every genuine nearby player was either taller or more
    # elongated than that).
    player_min_height_px: float = 16.0
    player_min_aspect: float = 1.8   # height / width — used by the soft ball OR
    # Positive human-proportion gate (stricter than the ball soft-reject alone).
    # A tagged body must be tall enough AND taller-than-wide within a person-like
    # aspect band. Spare balls, pitch-side cameras, and graphics are typically
    # near-square (aspect ≈ 1) or tiny — they clear the old AND-ball check when
    # slightly over ``player_min_height_px``, then win nearest-player. These
    # bounds keep ordinary standing / slightly crouched players and drop the
    # junk. Recall-safe: disabled when min_aspect ≤ 0.
    player_human_min_aspect: float = 1.30   # h/w floor — cameras/balls ~1.0
    player_human_max_aspect: float = 5.50   # poles / goal-post fragments
    # Soft ball-vs-player reject: when a "player" box overlaps this proxy ball
    # bbox (square half-size around the contact point), reject if it fails
    # EITHER height OR aspect (overlap is the extra evidence). Without overlap,
    # the strict BOTH check still applies.
    contact_ball_proxy_half_px: float = 8.0
    contact_ball_iou_soft: float = 0.25
    # Sideline / bench / coach reject: a "player" whose feet sit on dirt, track,
    # or concrete rather than pitch grass is not contesting the ball — drop them
    # before attribution. Recall-safe: only fires when the feet sample is
    # readable AND clearly non-grass; unmeasurable → keep.
    sideline_reject_enabled: bool = True
    sideline_min_feet_grass: float = 0.18
    # Soft official-kit fallback for referees the model mislabels as ``player``.
    # Typical ref kits (black / neon yellow) that match NEITHER team centroid are
    # labelled official and dropped. Never overrides a clear your-team match.
    official_kit_reject_enabled: bool = True

    # --- Crowded contacts (corner kicks, goalmouth scrambles) ---
    # When several players are packed around the ball, "nearest player" and
    # torso colour are both unreliable — the box we attribute the touch to may
    # simply be the wrong body, and its kit may be the wrong team's. Rather than
    # guess (and silently drop your real touch), flag the contact as crowded and
    # route it to the human: crowded touches are never auto-hidden and never
    # dropped by the colour gate, and they always appear in the review queue.
    crowd_detect_enabled: bool = True
    # Players within this distance of the ball count as contesting it. ~90px on
    # a 640-wide frame is roughly the width of a corner-kick cluster; must stay
    # under ``player_roi_half_px`` or the ROI itself bounds the count.
    crowd_radius_px: float = 90.0
    # This many contesting players ⇒ attribution is unreliable. 4 keeps ordinary
    # 1v1 and 2v2 duels on the normal path; corners/scrambles trip it.
    crowd_min_players: int = 4
    # Force crowded contacts into the review queue regardless of confidence, and
    # keep them out of the ``max_review`` cap so a long queue can't bury them.
    crowd_force_review: bool = True
    # Never drop a crowded contact as "other team" — in a pack the colour read
    # belongs to whichever body won the nearest-player race, not necessarily the
    # toucher.
    crowd_keep_other_team: bool = True
    # Own review budget for crowded touches (0 → unlimited). Bounds the worst
    # case on scrappy footage while still guaranteeing them a slot.
    crowd_max_review: int = 40
    # A count of nearby bodies answers the wrong question. Measured on a real
    # 5-minute run, 31 of 73 review clips were crowd-forced, 12 of them sitting
    # exactly on the count of 4 — while the box heights showed the 90px radius
    # was really only ~6m at that depth. Bodies being present is not the same as
    # attribution being contested: require that the runner-up is actually close
    # behind the winner before calling a contact crowded.
    crowd_require_ambiguity: bool = True
    # How far back the runner-up must be for the winner to count as clear, in
    # multiples of the winning box's own height (~1.75m of person). Expressed in
    # box heights rather than pixels so it means the same thing near and far
    # from the camera, which a fixed pixel gap does not.
    crowd_ambiguous_gap_heights: float = 0.8

    # --- Stage 6: per-contact team colour filter ---
    # 1 → 3 frames per contact (centre ±1). Each frame is a player-detection call,
    # so this directly scales Stage 5-6 cost; the jersey median over 3 frames is
    # robust enough.
    contact_color_window: int = 1   # frames each side of the contact to sample jersey
    contact_color_step: int = 1     # window step, in analysed-frame units
    team_color_max_dist: float = 60.0  # hue-weighted HSV distance ≤ this ⇒ your team
    # Conservative "clearly the other team" cutoff (grass-masked colours). A
    # contact whose kit is THIS far from the seed kit is dropped even when the
    # aggressive team_filter is off — the wide margin above team_color_max_dist
    # means only obviously-different kits (e.g. dark team, ref, keeper) are
    # dropped, never a same-team touch that merely reads noisy. Set high because
    # the seed player's own touches ranged up to ~95 on real footage.
    contact_other_team_dist: float = 115.0
    # Opponent-aware tie-break for the ambiguous colour band. When the opponent's
    # kit colour is known (from the team picker) AND the two kits are actually
    # distinguishable, a touch in the undecided band whose colour is closer to
    # the opponent centroid than to yours by at least this margin is dropped as
    # the other team. Uses BOTH team centroids instead of only "far from mine",
    # so obvious wrong-team touches in the 60-115 band stop reaching review.
    # Only ever acts inside the already-undecided band — never overrides a clear
    # "your team", so it can't cost you a real touch.
    contact_opponent_margin: float = 20.0
    # Min torso crop area (px) to trust a jersey colour. On wide footage tiny
    # crops are grass-contaminated; below this the contact is left undecided
    # (kept) rather than mislabelled. Real-frame debug: 10-30px-tall players give
    # ~6x12px torsos, so ~50px is a sane floor. Retune per footage.
    color_min_torso_px: int = 50
    # Master switch for the colour team-filter. Default OFF: PolyFut's footage is
    # typically wide amateur/phone video where torso colour is grass-contaminated
    # and can't separate teams (real-frame debug: even the target fails to match
    # its own seed), so filtering risks silently dropping real touches — and
    # sampling jersey colour across frames triples the Stage 5-6 detection cost.
    # Enable only for clean, broadcast-quality footage with distinct kits.
    # Issue 2's two-centroid gate on the single contact frame is the default path.
    team_filter_enabled: bool = False
    # Batch size for sparse player inference across nearby contact candidates
    # (Issue 7). 1 → one model call per contact (old behaviour). Larger batches
    # share a single Ultralytics/OpenVINO predict when the colour-window is off
    # (the default). Cap keeps memory bounded on 90-minute files.
    player_batch_size: int = 8

    # --- Stage 7: target scoring (appearance x orbital, sequential) ---
    # Appearance
    appearance_default: float = 0.5    # neutral score when appearance is unmeasurable
    orbital_anchor_min: float = 0.6    # appearance sim needed to (re)anchor the orbital
    # Orbital motion prior (pixel space; camera-comp/homography plug in via transform)
    orbital_base_px: float = 80.0      # radius at zero time gap
    orbital_growth_px_s: float = 60.0  # radius growth per second of gap
    orbital_max_gap_sec: float = 8.0   # beyond this the orbital covers the pitch → neutral
    orbital_floor: float = 0.5         # min prior — boost/tie-break only, never reject
    orbital_falloff: float = 0.5       # penalty slope outside the orbital
    # Tracklets
    tracklet_max_gap_sec: float = 3.0  # contacts within this window always share a tracklet
    # Colour-locked continuity (Issue 4): keep the same tracklet / orbital chain
    # across gaps up to this limit when the contact stays inside the orbital AND
    # kit colour is compatible with the anchor — same pattern as seed-clip
    # rejoin. Defaults to orbital_max_gap_sec so motion can carry identity past
    # the hard 3s cut without inventing links across half the match.
    continuity_max_gap_sec: float = 8.0
    continuity_color_max_dist: float = 60.0  # kit lock for chain rejoin
    # When a contact sits firmly inside the orbital on an active chain, weak /
    # ambiguous appearance must not drag confidence below this floor × prior.
    # Inverts "appearance leads, motion nudges" for well-anchored same-kit
    # sequences. Still well below autoaccept; never a hard reject path.
    motion_carry_floor: float = 0.70
    # Auto accept/hide thresholds (consumed by the Step 5 montage)
    autoaccept_conf: float = 0.85
    autohide_conf: float = 0.15
    # Auto-accept only when Stage 7 marks the contact as identity-linked to the
    # seeded player (orbital / motion continuity). High appearance on a random
    # same-kit body after a long gap must reach human review — otherwise it
    # becomes a false hotspot and merges into a long "you" zone.
    autoaccept_require_identity: bool = True
    # --- EXPERIMENT (branch: no-fallback-experiment) ---
    # How accurate can this be without leaning on the review queue? Measured on
    # a real 5-minute run: 73 of 88 contacts went to review and 0 were ever
    # auto-hidden, because confidence cannot fall below
    # appearance_default(0.5) x orbital_floor(0.5) = 0.25 while autohide_conf is
    # 0.15 — the auto-hide path is arithmetically unreachable. The pipeline can
    # say "definitely you" or "I don't know", never "definitely not you".
    #
    # Rather than lower the threshold (which would hide *unmeasurable* touches —
    # the one thing recall-safety forbids) a contact is hidden only on positive,
    # measured evidence: see scoring.negative_evidence_for. Set False to restore
    # the old behaviour exactly.
    autohide_on_evidence: bool = True
    # Appearance similarity below which a *successful* read counts as a positive
    # mismatch. Only applies when appearance was actually measurable; None/
    # unreadable never qualifies. Well under orbital_anchor_min (0.6) so it means
    # "clearly wrong", not "not clearly right".
    appearance_reject_max: float = 0.35
    # Hard cap on the review queue: only the top-N highest-confidence "review"
    # clips are shown; the rest are auto-hidden. Bounds the human review burden
    # even when appearance can't discriminate (low-res footage → most touches
    # land in the ambiguous middle). The montage is confidence-ranked, so these
    # are the most-likely-you clips.
    max_review: int = 80

    # --- Stage 8: review montage ---
    montage_clip_pad_sec: float = 1.0    # ±1s around the contact → ~2s review clips
    montage_crop_half_px: float = 120.0  # zoom half-size (resized px) around the contact

    # --- Adaptive review grouping (Stage 8 UX) ---
    # Tag each review touch with a kit-colour group + (within your team) an
    # appearance group, so "Not me" on the other team clears them all, and same-
    # kit decisions propagate softly. Cheap: reuses the contact torso crops.
    # Kit distance from your seed beyond this ⇒ confidently the *other team*
    # (hard-removable). Aligned with ``contact_other_team_dist`` so review UI
    # flags and the detection gate agree on what counts as opponent.
    grouping_other_team_dist: float = 115.0
    grouping_kit_cluster_dist: float = 55.0    # colour-clustering radius
    grouping_appearance_min_sim: float = 0.62  # same-kit appearance grouping (soft)

    # --- Stage 9: hotspot assembly (v1 semantics; pad widened to ±2s per doc) ---
    hotspot_pad_before_sec: float = 2.0
    hotspot_pad_after_sec: float = 2.0
    hotspot_gap_merge_sec: float = 5.0
    hotspot_min_zone_sec: float = 3.0
    # Run a hotspot on until the ball is seen to leave, instead of stopping a
    # fixed pad after the last *detected* touch. Touches are discrete; possession
    # is continuous, and at 40% ball-detection the last seen touch is usually not
    # the last touch. Measured on job fdc541cf493e: the final hotspot ended at
    # 290.8s = 288.8 + pad_after, cutting the player's last pass.
    hotspot_possession_enabled: bool = True
    # How far the ball must travel from where it sat at the last touch before it
    # counts as gone, in analysed-frame px. ~120px on a 640-wide frame is several
    # metres near the camera and more at distance — deliberately generous, since
    # over-running a clip by a second costs far less than truncating a goal.
    hotspot_possession_leave_px: float = 120.0
    # Cap on the extension. A ball never seen to leave is a tracking failure, not
    # a long dribble, so the window must not follow it indefinitely.
    hotspot_possession_max_extend_sec: float = 6.0

    # --- Reliability guardrail ---
    # A trajectory with almost no real detections means the ball model can't
    # see the ball (e.g. COCO yolov8 on a tiny/distant soccer ball). Per the
    # design doc, ball recall is the reliability ceiling and a degenerate
    # trajectory silently starves every downstream stage — so warn loudly.
    min_detected_ratio_warn: float = 0.2

    # --- Output ---
    output_dir: Path = Path("output_v2")

    # Every threshold whose unit is pixels of a ``target_width``-wide frame.
    # Changing the analysis width without scaling these silently redefines all
    # of them — an 80px "nearest player" gate means half as much pitch at 1280
    # as at 640 — so the two must never move independently.
    _PX_SCALED_FIELDS = (
        "roi_half_px", "ball_max_jump_px", "ball_suspect_jump_px",
        "ball_confirm_dist_px", "ball_pingpong_min_step_px",
        "camera_motion_max_shift_px", "camera_motion_exclude_half_px",
        "ball_surface_check_half_px", "contact_min_speed_px_s",
        "player_min_height_px", "player_roi_half_px",
        "contact_max_player_dist_px", "contact_metric_override_px",
        "contact_ball_proxy_half_px", "crowd_radius_px", "orbital_base_px",
        "orbital_growth_px_s", "montage_crop_half_px",
        "hotspot_possession_leave_px",
    )
    # Deliberately NOT scaled despite the ``_px`` name: the calibration screen
    # serves frames at the video's native resolution, so a landmark residual is
    # in click-frame pixels and has no relationship to the analysis width.
    _PX_NOT_FRAME_RELATIVE = ("calibration_max_median_px",)
    # Areas, not lengths — these scale with the square of the width.
    _PX_AREA_FIELDS = ("color_min_torso_px",)

    def for_frame_width(self, source_width: int | None):
        """This config re-expressed for the width a given source will run at.

        Returns ``self`` unchanged whenever the width does not move, so the
        common case (640-wide footage, which is every clip currently on disk)
        is provably identical to not calling this at all.

        Ratios, angles, fractions, seconds and counts are deliberately *not*
        scaled: an aspect ratio or a 0.6 similarity means the same thing at any
        resolution. Only lengths and areas measured in frame pixels move.
        """
        if not self.analysis_follow_source or not source_width:
            return self
        width = min(int(source_width), int(self.analysis_max_width))
        if width <= 0 or width == self.target_width:
            return self
        s = width / float(self.target_width)
        from dataclasses import replace
        changes = {f: getattr(self, f) * s for f in self._PX_SCALED_FIELDS
                   if hasattr(self, f)}
        for f in self._PX_AREA_FIELDS:
            if hasattr(self, f):
                changes[f] = type(getattr(self, f))(getattr(self, f) * s * s)
        changes["target_width"] = width
        return replace(self, **changes)

    def v1_config(self):
        """Project shared knobs onto a v1 ``PipelineConfig`` for the reused
        Stage 2 / Stage 3a code paths."""
        from polyfut_video.config import PipelineConfig

        return PipelineConfig(
            target_width=self.target_width,
            shot_filter_sample_every_n=self.shot_filter_sample_every_n,
            cut_hist_threshold=self.cut_hist_threshold,
            green_ratio_min=self.green_ratio_min,
            motion_smooth_max=self.motion_smooth_max,
            graphic_uniform_ratio=self.graphic_uniform_ratio,
            deadtime_motion_threshold=self.deadtime_motion_threshold,
            deadtime_min_duration_sec=self.deadtime_min_duration_sec,
        )


DEFAULT_CONFIG = PipelineV2Config()
