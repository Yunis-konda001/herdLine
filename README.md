# HerdLine

HerdLine counts goats in a gate video. A farmer signs in, uploads a short clip, types the herd size, and sees how many goats crossed a line.

## Links

- GitHub: https://github.com/Yunis-konda001/herdLine
- Demo video: https://youtu.be/s_iRQ79fcIs
- Public test URL: see [docs/DEPLOY.md](docs/DEPLOY.md) (tunnel or server)

## What is in this project

| Path | What it is |
| --- | --- |
| `notebooks/goat_detection_and_counting.ipynb` | Data, training notes, detection scores |
| `models/weights/best.pt` | Fine-tuned YOLOv8n goat detector |
| `src/` | Detect, track, count, and save results |
| `web/` | Website (sign-up, check-in, history) |
| `config/bytetrack.yaml` | Tracker settings for flickering boxes |
| `docs/DEPLOY.md` | How to share a public test link |
| `docs/screenshots/` | UI screenshots for the report |

The CherryChevre images in `data/train`, `data/valid`, and `data/test` stay on your computer (~9 GB). They are not uploaded to GitHub. `data/goat.yaml` is in the repo.

## Setup

```bash
cd c:/capstone
.venv\Scripts\activate
pip install -r requirements.txt
```

You need Python 3 and the packages in `requirements.txt`.

## Run the website

```bash
uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

Demo steps:

1. Sign up (email, password, farm name, herd size).
2. New check-in — enter herd size and upload a gate video.
3. Summary — counted goats, missing or extra, annotated video.
4. History — check-ins for that account on this browser.
5. Log out when finished.

## Notebook

Open `notebooks/goat_detection_and_counting.ipynb`.

It explains the dataset, preprocessing, YOLOv8n, and the test-set scores.

On the held-out test images the saved model scored about:

| Metric | Score |
| --- | ---: |
| Precision | 0.90 |
| Recall | 0.74 |
| F1 | 0.81 |
| mAP50 | 0.82 |

The training cell has already been run. Do not run it again unless you want to replace `best.pt`.

## Screenshots

![Farm profile](docs/screenshots/profile.png)

![New check-in](docs/screenshots/check-in.png)

![Summary](docs/screenshots/summary.png)

![History](docs/screenshots/history.png)

## How a count is made

1. Each video frame goes through YOLOv8n (goat boxes).
2. ByteTrack keeps a temporary ID on each goat.
3. Track memory keeps that ID when the box flickers for a few frames.
4. The line counter counts a goat once on the entry side of the line.
5. The total is compared with the herd size the farmer typed.

## Public testing

For your defense, share a temporary public URL while the laptop runs the app.
Follow [docs/DEPLOY.md](docs/DEPLOY.md).
