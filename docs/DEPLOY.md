# How to share HerdLine with a public URL

You need a link that a marker or tester can open from another computer.
Vercel and Netlify cannot run this app (YOLO needs a long-running server).

Pick one option below. For a **project defense**, Option A is the fastest.

---

## Option A — Temporary public link (best for defense day)

Keep the app running on your laptop. Create a short public tunnel.

### 1. Start HerdLine on your machine

```bash
cd c:/capstone
.venv\Scripts\activate
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Check that http://127.0.0.1:8000 works first.

### 2. Make a public URL with Cloudflare Tunnel (free)

1. Install cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/
2. In a **second** terminal, run:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

3. Copy the `https://....trycloudflare.com` link that appears.
4. Send that link to the tester.
5. Leave **both** terminals open while they test.

### Or use ngrok (also free)

```bash
ngrok http 8000
```

Use the `https://....ngrok-free.app` link.

**Important for defense**

- Keep your laptop plugged in and awake.
- Do not close the uvicorn or tunnel windows.
- Test the link yourself on a phone before the defense.
- Ask the tester to upload a **short** video (a few seconds). Long clips take longer to count.

---

## Option B — Always-on server (better after defense)

Use a small Linux virtual machine (Oracle Cloud free tier, or any cheap VPS with **at least 4 GB RAM**).

1. Copy the project and `models/weights/best.pt` to the server.
2. Install Python 3 and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

3. Put Nginx or Caddy in front for HTTPS.
4. Raise the proxy timeout to **at least 5 minutes** so a long video is not cut off.
5. Put the public URL in the README.

You do **not** need the 9 GB training images on the server. You only need `best.pt`.

---

## What the tester should do

1. Open `/login` and create an account (or sign in).
2. Go to **New check-in**.
3. Enter the herd size and upload a short gate video.
4. Wait for the summary and the annotated video.
5. Check **History** for that account only.

---

## What not to use

- Vercel / Netlify — request time and package size are too small for YOLO.
- Streamlit Cloud — would need a full rewrite of this website.
