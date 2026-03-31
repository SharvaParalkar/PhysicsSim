# PhysicsSim

Genesis-based particle drop simulation with a FastAPI websocket backend and a Three.js frontend viewer.

## Prerequisites

- Python 3.10+ (Windows PowerShell examples below)
- Node.js 18+ and npm
- A compatible GPU driver if you want Genesis GPU mode

## 1) Backend setup (PowerShell)

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Frontend setup

In a second terminal:

```powershell
cd frontend
npm install
```

## 3) Run the app

Use two terminals at the same time.

Terminal A (backend, from project root):

```powershell
.\.venv\Scripts\Activate.ps1
python server.py
```

Terminal B (frontend):

```powershell
cd frontend
npm run dev
```

Then open the Vite URL (usually [http://localhost:5173](http://localhost:5173)).

## Optional: Run simulation script directly

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python simulation.py --backend auto --n 100 --duration 5.0
```

Outputs are written to `results/` (`particles.csv`, `contact_pairs.csv`, `contact_points.csv`, `results.h5`).

## Troubleshooting

- If backend fails on GPU init, it automatically falls back to CPU.
- If PowerShell blocks activation scripts, run:
  - `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
