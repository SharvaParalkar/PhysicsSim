import asyncio
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

import h5py
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

import simulation


app = FastAPI()


@app.get("/results")
def get_results() -> Dict[str, List[Dict[str, Any]]]:
  output_dir = simulation.OUTPUT_DIR
  particles_path = os.path.join(output_dir, "particles.csv")
  pairs_path = os.path.join(output_dir, "contact_pairs.csv")
  points_path = os.path.join(output_dir, "contact_points.csv")

  if not os.path.exists(particles_path):
    raise HTTPException(status_code=404, detail="No simulation results found. Run a simulation first.")

  particles = pd.read_csv(particles_path).to_dict(orient="records")
  contact_pairs = pd.read_csv(pairs_path).to_dict(orient="records") if os.path.exists(pairs_path) else []
  contact_points = pd.read_csv(points_path).to_dict(orient="records") if os.path.exists(points_path) else []

  return {
      "particles": particles,
      "contact_pairs": contact_pairs,
      "contact_points": contact_points,
  }


@app.get("/metrics")
def get_metrics() -> Dict[str, Any]:
  output_dir = simulation.OUTPUT_DIR
  h5_path = os.path.join(output_dir, "simulation.h5")

  if not os.path.exists(h5_path):
    raise HTTPException(status_code=404, detail="No simulation metrics found. Run a simulation first.")

  with h5py.File(h5_path, "r") as f:
    return {
        "Z": float(f.attrs.get("Z", 0.0)),
        "total_pp": int(f.attrs.get("total_pp", 0)),
        "total_pc": int(f.attrs.get("total_pc", 0)),
        "n_isolated": int(f.attrs.get("n_isolated", 0)),
        "n_container_touch": int(f.attrs.get("n_container_touch", 0)),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
  await ws.accept()
  try:
    while True:
      msg = await ws.receive_text()
      if msg.strip().lower() == "start":
        proc = subprocess.Popen(
            [sys.executable, "simulation.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
          await ws.send_text(line.rstrip("\n"))
        await ws.send_text("SIMULATION_DONE")
      else:
        await ws.send_text(json.dumps({"error": "Unknown command"}))
  except WebSocketDisconnect:
    return


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

