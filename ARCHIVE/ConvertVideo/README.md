# webm → mp4 (local converter)

A tiny local web app: a Node server that runs `ffmpeg`/`ffprobe` on your machine,
with a browser page as the UI. Browsers can't shell out to a terminal directly,
so this runs a small localhost server that does the actual conversion — the
page just tells it what to run and streams the log back live.

## Setup (one-time)

1. Make sure `ffmpeg` and `ffprobe` are on your PATH (you already have ffmpeg
   working from PowerShell, and ffprobe ships alongside it in the same folder —
   confirm with `ffprobe -version`).
2. Install Node.js if you don't have it: https://nodejs.org
3. Open this folder in PowerShell and run:
   ```
   npm install
   ```

## Run it

```
npm start
```

Then open **http://localhost:4173** in your browser.

## Use it

**Option A — drag and drop:**
1. Drag your `.webm` file onto the dropzone (or click it to browse). It
   uploads to the local server (browsers won't reveal a real filesystem path
   from a drag-and-drop, so this is the only way a browser page can hand the
   file to ffmpeg).
2. FPS is auto-detected right after upload.
3. Click **Convert**. When it finishes, click **Download converted mp4** in
   the log to save it whereever you like.

**Option B — point at a file already on disk:**
1. Paste the full path (e.g. `C:\Users\sharv\Downloads\starinterlock.webm`)
   into the path field instead.
2. Click **Detect FPS**, then **Convert**. The output is saved next to the
   input with the same name — no upload/download round trip needed.

Either way, FPS detection runs `ffprobe` and computes the real frame rate from
frame count ÷ duration, which fixes the bogus ~1000fps timebase Chrome
sometimes writes into recorded webm files (the cause of the
`malloc of size ... failed` / `MB rate > level limit` error).

## Notes

- If your webm has an alpha channel (transparency), it will be dropped —
  H.264/mp4 can't carry alpha. Say the word if you'd rather output to
  something that preserves it (e.g. ProRes 4444 in a .mov), and I'll add a
  toggle for that.
- Everything runs locally; no files leave your machine.
