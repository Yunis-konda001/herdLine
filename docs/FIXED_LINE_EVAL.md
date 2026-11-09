# Fixed-line counting evaluation

## Method

- Pipeline: `CountingPipeline` (same path as the web app)
- Fixed vertical line at `x = 0.5`
- Direction: `either` (counts L→R and R→L; walk direction does not zero the count)
- **No** herd-size line / direction search
- Herd size / hand count used only afterward for error, misses, extras, accuracy
- Script: `python scripts/eval_fixed_line.py`
- Raw CSV: `outputs/results/fixed_line_eval.csv`

## Results

| Video | Expected | System | Error | Misses | Extras | Accuracy | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_video_4_goats.mp4 | 4 | 3 | -1 | 1 | 0 | 0.750 | 14.8 |
| test_video_5_goats.mp4 | 5 | 4 | -1 | 1 | 0 | 0.800 | 8.3 |
| test_video_6_goats.mp4 | 6 | 6 | +0 | 0 | 0 | 1.000 | 18.5 |
| test_video_8_goats.mp4 | 8 | 6 | -2 | 2 | 0 | 0.750 | 42.9 |
| test_video_12_goats.mp4 | 12 | 8 | -4 | 4 | 0 | 0.667 | 84.3 |
| test_video_17_goats.mp4 | 17 | 5 | -12 | 12 | 0 | 0.294 | 42.3 |

Mean accuracy: **0.710**.

## Notes for the report

- Counting is independent of declared herd size (no line fitting).
- Default direction `either` so opposite walk clips are not zeroed.
- Dense / crowded gates (especially 17) still undercount; image mAP does not equal video counting accuracy.
