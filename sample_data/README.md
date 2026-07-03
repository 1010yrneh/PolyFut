# sample_data

Drop a short test clip here (e.g. `clip.mp4`) to validate the pipeline.
Video files are gitignored; the label/template files below are tracked.

- `labels.template.json` — copy to `labels.json` and hand-label true possession
  windows for your clip (Session 0b). Used to measure precision/recall.
- `angle_ranges.example.yaml` — optional camera-angle ranges for the ball-recall
  spike, so you can see recall per angle (elevated vs sideline).
