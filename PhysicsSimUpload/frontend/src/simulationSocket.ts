/**
 * Single shared WebSocket for the simulation API.
 * Avoids "closed before connection" when React Strict Mode mounts/unmounts/remounts.
 */

const DEFAULT_WS = 'ws://localhost:8000/ws'

export function getSimulationWsUrl(): string {
  const v = import.meta.env.VITE_WS_URL as string | undefined
  if (typeof v === 'string' && v.length > 0) return v
  return DEFAULT_WS
}

let shared: WebSocket | null = null

export function getSharedSimulationSocket(): WebSocket {
  if (shared && (shared.readyState === WebSocket.OPEN || shared.readyState === WebSocket.CONNECTING)) {
    return shared
  }
  shared = new WebSocket(getSimulationWsUrl())
  return shared
}
