// dashboard/api/control.js — on/off switch for the autonomous runner.
//
// State lives ON THE ALPACA PAPER ACCOUNT as a watchlist: the runner treats
// the existence of the watchlist "FVG-COPILOT-ENABLED" as "trading ON".
// Pressing the button creates/deletes that watchlist via the Alpaca API —
// no extra database, no new service, and even the kill switch runs through
// Alpaca infrastructure.
//
//   GET  /api/control            -> { "enabled": true|false }
//   POST /api/control {enabled}  -> creates/deletes the watchlist, returns new state
//
// The runner polls this state at most every scan cycle (~5 min) and gates NEW
// entries only — risk enforcement (stop-loss / hold caps) always runs.
//
// ENV: must match whatever dashboard/api/status.js already uses. Supports
// both common env var naming schemes.

const API_BASE = "https://paper-api.alpaca.markets";
const WATCHLIST_NAME = "FVG-COPILOT-ENABLED";

const KEY = process.env.ALPACA_API_KEY || process.env.APCA_API_KEY_ID;
const SECRET = process.env.ALPACA_SECRET_KEY || process.env.APCA_API_SECRET_KEY;

const headers = {
  "APCA-API-KEY-ID": KEY,
  "APCA-API-SECRET-KEY": SECRET,
};

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  if (req.method === "OPTIONS") return res.status(204).end();

  if (!KEY || !SECRET) {
    return res.status(500).json({ error: "env_not_set" });
  }

  try {
    const listsResp = await fetch(`${API_BASE}/v2/watchlists`, { headers });
    if (!listsResp.ok) {
      return res.status(502).json({ error: `alpaca_${listsResp.status}` });
    }
    const lists = await listsResp.json();
    const existing = (Array.isArray(lists) ? lists : []).find(
      (w) => w.name === WATCHLIST_NAME
    );

    if (req.method === "GET") {
      return res.status(200).json({ enabled: !!existing });
    }

    if (req.method === "POST") {
      const enabled = !!(req.body && req.body.enabled);

      if (enabled && !existing) {
        const createResp = await fetch(`${API_BASE}/v2/watchlists`, {
          method: "POST",
          headers: { ...headers, "Content-Type": "application/json" },
          body: JSON.stringify({ name: WATCHLIST_NAME, symbols: ["SPY"] }),
        });
        if (!createResp.ok) {
          return res.status(502).json({ error: `alpaca_create_${createResp.status}` });
        }
      } else if (!enabled && existing) {
        const delResp = await fetch(`${API_BASE}/v2/watchlists/${existing.id}`, {
          method: "DELETE",
          headers,
        });
        if (!delResp.ok) {
          return res.status(502).json({ error: `alpaca_delete_${delResp.status}` });
        }
      }

      return res.status(200).json({ enabled });
    }

    return res.status(405).json({ error: "method_not_allowed" });
  } catch (e) {
    return res.status(502).json({ error: String(e) });
  }
}
