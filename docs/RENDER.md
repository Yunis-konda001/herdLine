# Deploy HerdLine on Render (ONNX)

Use the slim stack: `best.onnx` + ONNX Runtime — **no PyTorch** on the server.

## Render settings

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Build command | `pip install -r requirements-render.txt` |
| Start command | `uvicorn web.app:app --host 0.0.0.0 --port $PORT` |
| Env `HERDLINE_IMGSZ` | `640` |
| Env `HERDLINE_CONF` | `0.18` |
| Plan | Prefer **≥2 GB RAM** (512 MB still may fail) |

## Required files on the service

- `web/`, `src/`, `config/`
- `models/weights/best.onnx` (not the full training set)
- `requirements-render.txt`

Do **not** install `ultralytics` / `torch` on Render.

## After deploy

Upload **one short** gate video at a time. Delete old files under `outputs/` if disk grows.
