const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 4173;

const UPLOAD_DIR = path.join(__dirname, 'uploads');
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    // Prefix with a timestamp so repeat drops of same-named files don't collide.
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});
const upload = multer({ storage, limits: { fileSize: 5 * 1024 * 1024 * 1024 } }); // 5GB cap

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Serve the UI explicitly at root, independent of static-file resolution quirks.
app.get('/', (req, res) => {
  const indexPath = path.join(__dirname, 'public', 'index.html');
  if (!fs.existsSync(indexPath)) {
    return res
      .status(500)
      .send(
        `public/index.html not found. Expected it at: ${indexPath}\n` +
          `Make sure the "public" folder sits next to server.js.`
      );
  }
  res.sendFile(indexPath);
});

// Chrome (129+) auto-probes this on every localhost page. Answer quietly
// instead of letting it 404/CSP-error in the console.
app.get('/.well-known/appspecific/com.chrome.devtools.json', (req, res) => {
  res.status(204).end();
});

// ---- Helpers -------------------------------------------------------------

function runCapture(cmd, args) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { windowsHide: true });
    let out = '';
    let err = '';
    p.stdout.on('data', (d) => (out += d.toString()));
    p.stderr.on('data', (d) => (err += d.toString()));
    p.on('error', (e) => reject(e));
    p.on('close', (code) => {
      if (code === 0) resolve(out || err);
      else reject(new Error(err || `Exited with code ${code}`));
    });
  });
}

function deriveOutputPath(inputPath) {
  const dir = path.dirname(inputPath);
  const base = path.basename(inputPath, path.extname(inputPath));
  return path.join(dir, `${base}.mp4`);
}

// Compute the real fps from frame count / duration. webm/mkv containers often
// only report duration at the container (format) level, not per-stream, so
// we ask ffprobe for both and fall back to whichever is present.
async function detectFps(inputPath) {
  const raw = await runCapture('ffprobe', [
    '-v', 'error',
    '-count_frames',
    '-select_streams', 'v:0',
    '-show_entries', 'stream=nb_read_frames,duration:format=duration',
    '-of', 'json',
    inputPath,
  ]);
  const parsed = JSON.parse(raw);
  const stream = parsed.streams && parsed.streams[0];
  const frames = stream ? parseFloat(stream.nb_read_frames) : NaN;
  const streamDuration = stream ? parseFloat(stream.duration) : NaN;
  const formatDuration = parsed.format ? parseFloat(parsed.format.duration) : NaN;
  const duration = !isNaN(streamDuration) && streamDuration > 0 ? streamDuration : formatDuration;

  let fps = null;
  if (frames && duration) {
    fps = Math.round((frames / duration) * 100) / 100;
  }
  return { frames, duration, fps };
}

// ---- Routes ---------------------------------------------------------------

// Receive a dragged/dropped file's bytes and save to the local uploads folder,
// since browsers don't expose a real filesystem path from drag-and-drop.
app.post('/upload', upload.single('file'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file received.' });
  res.json({ path: req.file.path, name: req.file.originalname });
});

// Send a converted (or original) file back to the browser as a download.
app.get('/download', (req, res) => {
  const filePath = req.query.path;
  if (!filePath || !fs.existsSync(filePath)) {
    return res.status(404).send('File not found.');
  }
  res.download(filePath);
});

// Detect real fps using ffprobe (frame count / duration), since Chrome-recorded
// webm files often report a bogus 1000fps timebase that breaks libx264.
app.get('/detect', async (req, res) => {
  const inputPath = req.query.path;
  if (!inputPath || !fs.existsSync(inputPath)) {
    return res.status(400).json({ error: 'File not found at that path.' });
  }
  try {
    const result = await detectFps(inputPath);
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Convert with Server-Sent Events streaming ffmpeg's stderr log live to the page.
app.get('/convert', async (req, res) => {
  const inputPath = req.query.path;
  let fps = req.query.fps;

  if (!inputPath || !fs.existsSync(inputPath)) {
    res.writeHead(400, { 'Content-Type': 'text/event-stream' });
    res.write(`event: error\ndata: File not found at that path.\n\n`);
    return res.end();
  }

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });

  const send = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  // Backstop: never run ffmpeg without a real fps value. A missing/blank fps
  // is exactly what let the 1000fps timebase bug slip through before.
  if (!fps) {
    send('log', 'No fps supplied — auto-detecting before converting (backstop check).');
    try {
      const result = await detectFps(inputPath);
      if (result.fps) {
        fps = result.fps;
        send('log', `Auto-detected fps: ${fps}`);
      }
    } catch (e) {
      send('log', `fps auto-detection failed: ${e.message}`);
    }
  }

  if (!fps) {
    send(
      'error',
      'Could not determine fps and none was supplied — refusing to convert without it, since that causes the libx264 crash on high-res Chrome recordings.'
    );
    return res.end();
  }

  const outputPath = deriveOutputPath(inputPath);

  const args = [];
  if (fps) args.push('-r', String(fps));
  args.push('-i', inputPath, '-c:v', 'libx264', '-pix_fmt', 'yuv420p');
  if (fps) args.push('-r', String(fps));
  args.push('-c:a', 'aac', '-y', outputPath);

  send('log', `> ffmpeg ${args.join(' ')}`);

  const ff = spawn('ffmpeg', args, { windowsHide: true });

  ff.stderr.on('data', (d) => {
    d.toString()
      .split(/\r?\n/)
      .filter(Boolean)
      .forEach((line) => send('log', line));
  });

  ff.on('error', (e) => {
    send('error', e.message);
    res.end();
  });

  ff.on('close', (code) => {
    if (code === 0) {
      send('done', { outputPath });
    } else {
      send('error', `ffmpeg exited with code ${code}`);
    }
    res.end();
  });

  req.on('close', () => {
    if (!ff.killed) ff.kill();
  });
});

app.listen(PORT, () => {
  const indexPath = path.join(__dirname, 'public', 'index.html');
  if (!fs.existsSync(indexPath)) {
    console.warn(
      `\n⚠️  Warning: could not find ${indexPath}\n` +
        `   Make sure the folder layout is:\n` +
        `     webm2mp4/\n` +
        `       server.js\n` +
        `       package.json\n` +
        `       public/\n` +
        `         index.html\n`
    );
  }
  console.log(`\nwebm2mp4 running -> open http://localhost:${PORT} in your browser\n`);
});