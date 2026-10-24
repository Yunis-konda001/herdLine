# Public URL for testing (defense)

You need a link other people can open. Vercel and Netlify cannot run this app.

## Best option for defense day: tunnel from your laptop

### 1. Start the app

```bash
cd c:/capstone
.venv\Scripts\activate
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Check http://127.0.0.1:8000 first.

### 2. Make a public link (Cloudflare Tunnel)

In a second terminal:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Copy the `https://....trycloudflare.com` link.

Or use ngrok:

```bash
ngrok http 8000
```

### 3. Keep both windows open while people test

Ask them to upload a **short** video. Long clips take longer to count.

## Longer option: small Linux server

Use a VM with at least 4 GB RAM. Copy the project and `models/weights/best.pt`. You do not need the 9 GB training images.

```bash
pip install -r requirements.txt
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Put HTTPS in front if you can. Raise the proxy timeout to at least 5 minutes.
