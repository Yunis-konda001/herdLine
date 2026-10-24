# HerdLine

Counts goats in a short gate video. Sign in, upload a clip, enter the herd size, and see how many crossed the line.

- GitHub: https://github.com/Yunis-konda001/herdLine
- Demo: https://youtu.be/s_iRQ79fcIs
- Public test link: [docs/DEPLOY.md](docs/DEPLOY.md)

## Setup

```bash
cd c:/capstone
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

You need `models/weights/best.pt` (included). Training images are not in Git (~9 GB); only `data/goat.yaml` is.

## Run

```bash
uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 — sign up, start a check-in, upload a short gate video with the herd size.

## What’s in the repo

| Path | Role |
| --- | --- |
| `src/` | Detect, track, count |
| `web/` | FastAPI site |
| `models/weights/best.pt` | Fine-tuned YOLOv8n goat detector |
| `config/bytetrack.yaml` | Tracker settings |
| `data/videos/evaluation/` | Sample gate clips (hand counts in the filenames) |
| `notebooks/goat_detection_and_counting.ipynb` | Training / evaluation notes |
| `docs/DEPLOY.md` | How to share a public test URL |

## How counting works

1. YOLOv8n finds goats in each frame  
2. Weak / odd boxes are filtered out  
3. Tracks are kept across frames; a goat is counted **once** when its centre crosses the gate line  
4. If herd size is given, both walk directions are compared in **one** video pass, then the better direction is used for the annotated output  

Prefer short clips. Long videos take longer.

## Notebook

Open `notebooks/goat_detection_and_counting.ipynb` for preprocessing, the R-CNN baseline, YOLO training, and metrics. Run training cells only when you can leave the machine on.

## Author

Kumi Yunis Konda — BSc Software Engineering, African Leadership University  
Supervisor: Hubert Apana
