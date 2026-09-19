#!/usr/bin/env python3
"""
FVG Copilot — Autonomous Runner (v6: live agent reasoning + dual-source signals)
===================================================================
v6 changes vs v5:
  - REASONING EVENTS: decision / veto / order_submitted / pipeline_error
    activity events now carry the full agent reasoning chain — Scout's
    thesis, confidence score, suggested strategy, direction and underlying
    price, plus Risk Guardian's caps, rationale and veto reason, and the
    signal's source label (momentum_confirmed vs raw_gap from the
    dual-engine signal adapter). The dashboard's Agent Reasoning panel
    renders these live.
  - ACTIVITY BUFFER widened from 60 to 200 events (market-closed noise at
    ~12/hr was evicting decision events within ~5 hours).

v5 changes vs v4:
  - ACTIVITY FEED: an in-memory ring buffer records every scan-pass
    outcome (no signal, kill-switch off, market closed, clock-check
    error, VETO, APPROVE, ORDER_SUBMITTED, pipeline error) and every enforcer
    action (stop-loss hit, hold-cap hit, adopted orphan position). Exposed as
    GET /activity JSON on the same health-check HTTP server used for Render
    keep-alive. dashboard/api/status.js fetches this server-to-server and
    merges it in, so the dashboard shows what the bot is ACTUALLY doing
    right now instead of only the last Alpaca fill. Ephemeral by design
    (lost on restart) — same trade-off already accepted for agent_state.json
    on the free tier; this is a live window, not a permanent audit log
    (that remains STATE_DIR/runs/*.json).

v4 changes vs v3:
  - HEALTH CHECK HTTP SERVER: Render's free tier supports always-running
    WEB services (it's Background Workers that need a paid plan). The equity
    bot (live_bot.py) already uses this exact pattern — HTTPServer bound to
    PORT, which Render injects. An external free pinger (UptimeRobot /
    cron-job.org, every 5 min) prevents the 15-min spin-down, giving a
    24/7 runner on the free plan.
  - STATE IS EPHEMERAL on free tier (no persistent disk): on restart the
    runner boots with empty state. By design this is safe — the reconcile
    layer rebuilds from the live Alpaca account every pass and adopts any
    open option position with default caps. No position can ever be left
    unmanaged; worst case is a default cap instead of the Risk Guardian's
    custom one.

v3 changes vs v2:
  - KILL SWITCH: scan passes are gated on the existence of an Alpaca watchlist
    named KILLSWITCH_WATCHLIST (default "FVG-COPILOT-ENABLED") on the paper
    account. The dashboard's on/off button (dashboard/api/control.js) creates
    and deletes that watchlist. Watchlist present = trading ON; absent = OFF.
    Fail-closed: any API error during the check counts as OFF.
    The ENFORCER is never gated — risk enforcement (stop-loss, hold caps,
    reconcile) always runs, so "off" can never mean "unmanaged".
============================================================================
Single long-running process for one Render web service:

  [enforcer thread, ENFORCE_INTERVAL_SEC=60]
      Enforce Risk-Guardian caps (final_max_loss_usd / final_max_hold_hours)
      on open option positions. Closes via Alpaca's position-close endpoint.
      Reconciles against the live account every pass.

  [scanner thread, SCAN_INTERVAL_SEC=300, market-hours gated via /v2/clock]
      Kill-switch check -> market-hours check -> poll agents/signal_adapter.py
      (wraps live_bot.py's validated FVG detection) -> run the REAL
      pipeline.run_pipeline() end to end. Reasoning chains saved to
      STATE_DIR/runs/*.json.
"""
import argparse
import collections
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(ROOT))

LOG = logging.getLogger("runner")

CFG = {
    "STATE_DIR": Path(os.environ.get("STATE_DIR", ROOT)),
    "SCAN_INTERVAL_SEC": int(os.environ.get("SCAN_INTERVAL_SEC", "300")),
    "ENFORCE_INTERVAL_SEC": int(os.environ.get("ENFORCE_INTERVAL_SEC", "60")),
    "SYMBOLS": [s.strip().upper() for s in os.environ.get("SYMBOLS", "SPY,QQQ,XLV,XLF,IWM").split(",") if s.strip()],
    "DRY_RUN": os.environ.get("DRY_RUN", "true").lower() != "false",
    "MAX_DAILY_TRADES": int(os.environ.get("MAX_DAILY_TRADES", "4")),
    "DEFAULT_MAX_HOLD_HOURS": float(os.environ.get("DEFAULT_MAX_HOLD_HOURS", "24")),
    "ADOPT_UNKNOWN_POSITIONS": os.environ.get("ADOPT_UNKNOWN_POSITIONS", "true").lower() != "false",
    "SIGNAL_SOURCE": os.environ.get("SIGNAL_SOURCE", "live_bot"),  # live_bot | simulated | off
    "ESTIMATED_TRADE_COST_USD": float(os.environ.get("ESTIMATED_TRADE_COST_USD", "500")),
    "CLI_TIMEOUT_SEC": int(os.environ.get("CLI_TIMEOUT_SEC", "30")),
    "KILLSWITCH_WATCHLIST": os.environ.get("KILLSWITCH_WATCHLIST", "FVG-COPILOT-ENABLED"),
    "REQUIRE_KILLSWITCH_ON": os.environ.get("REQUIRE_KILLSWITCH_ON", "true").lower() != "false",
    "HEALTH_PORT": int(os.environ.get("PORT", "10000")),
    "DATABASE_URL": os.environ.get("DATABASE_URL", ""),
    "EVENT_RETENTION_HOURS": int(os.environ.get("EVENT_RETENTION_HOURS", "48")),
}

OCC_RE = re.compile(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$")
_state_lock = threading.Lock()
_stop = threading.Event()

_activity = collections.deque(maxlen=1000)
_activity_lock = threading.Lock()
_db_lock = threading.Lock()
_db_ready = False


def db_connect():
    if not CFG["DATABASE_URL"]:
        raise RuntimeError("DATABASE_URL is required for durable recovery state")
    return psycopg.connect(CFG["DATABASE_URL"], row_factory=dict_row)


def init_database():
    global _db_ready
    with _db_lock:
        if _db_ready:
            return
        with db_connect() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS runner_state (
                    state_key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS runner_events (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL,
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL,
                    extra JSONB
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS runner_events_ts_idx ON runner_events (ts DESC)")
        _db_ready = True
        LOG.info("durable Postgres state initialized")


def record_activity(kind: str, message: str, extra: dict = None):
    """Record live activity in memory and durable Postgres; never raises."""
    entry = {"ts": utcnow().isoformat(), "kind": kind, "message": message}
    if extra:
        entry["extra"] = extra
    try:
        with _activity_lock:
            _activity.appendleft(entry)
        if _db_ready:
            with db_connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO runner_events (ts, kind, message, extra) VALUES (%s, %s, %s, %s)",
                    (entry["ts"], kind, message, json.dumps(extra) if extra else None),
                )
                cur.execute(
                    "DELETE FROM runner_events WHERE ts < NOW() - (%s * INTERVAL '1 hour')",
                    (CFG["EVENT_RETENTION_HOURS"],),
                )
    except Exception as e:
        LOG.warning("activity persistence failed: %s", e)


def durable_events():
    if not _db_ready:
        with _activity_lock:
            return list(_activity)
    try:
        with db_connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT ts, kind, message, extra FROM runner_events "
                "WHERE ts >= NOW() - (%s * INTERVAL '1 hour') ORDER BY ts DESC LIMIT 2000",
                (CFG["EVENT_RETENTION_HOURS"],),
            )
            rows = cur.fetchall()
        return [
            {"ts": r["ts"].isoformat(), "kind": r["kind"], "message": r["message"],
             **({"extra": r["extra"]} if r["extra"] else {})}
            for r in rows
        ]
    except Exception as e:
        LOG.warning("durable event read failed: %s", e)
        with _activity_lock:
            return list(_activity)


# ------------------------------------------------------------ health server

class HealthHandler(BaseHTTPRequestHandler):
    """Minimal 200 OK on "/" for Render/UptimeRobot keep-alive, plus a real
    GET /activity JSON feed (kill-switch state, scan outcomes, VETOs, fills,
    errors, full agent reasoning) that dashboard/api/status.js merges into
    what the tiles show."""

    def _respond(self, status=200, body=b"OK", content_type="text/plain"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _respond_activity(self):
        try:
            events = durable_events()
            payload = json.dumps({
                "ok": True,
                "generated_at": utcnow().isoformat(),
                "killswitch_watchlist": CFG["KILLSWITCH_WATCHLIST"],
                "require_killswitch_on": CFG["REQUIRE_KILLSWITCH_ON"],
                "scan_interval_sec": CFG["SCAN_INTERVAL_SEC"],
                "events": events,
            }, default=str).encode("utf-8")
            self._respond(200, payload, "application/json")
        except Exception as e:
            payload = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
            self._respond(500, payload, "application/json")

    def do_GET(self):
        if self.path.rstrip("/") == "/activity":
            self._respond_activity()
        else:
            self._respond()

    def do_HEAD(self):
        self._respond()

    def log_message(self, format, *args):
        return


def start_health_server():
    server = HTTPServer(("0.0.0.0", CFG["HEALTH_PORT"]), HealthHandler)
    LOG.info("health check server on port %d (Render keep-alive + /activity)", CFG["HEALTH_PORT"])
    threading.Thread(target=server.serve_forever, name="health", daemon=True).start()


# ---------------------------------------------------------------- Alpaca CLI

class CliError(RuntimeError):
    pass


def alpaca_cli(args, body=None):
    cmd = ["alpaca"] + args
    try:
        proc = subprocess.run(
            cmd, input=body, capture_output=True, text=True,
            timeout=CFG["CLI_TIMEOUT_SEC"],
        )
    except FileNotFoundError:
        raise CliError("alpaca CLI not found on PATH — install per repo README")
    except subprocess.TimeoutExpired:
        raise CliError(f"alpaca CLI timeout: {' '.join(args)}")
    if proc.returncode != 0:
        raise CliError(f"alpaca CLI failed ({proc.returncode}): {proc.stderr.strip()[:400]}")
    out = proc.stdout.strip()
    if not out:
        return {}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"raw": out}


def api(method, path, body=None):
    return alpaca_cli(["api", method, path], body=json.dumps(body) if body is not None else None)


# ------------------------------------------------------------------- state

def default_state() -> dict:
    return {"trades": [], "daily": {"date": "", "count": 0}}


def load_state() -> dict:
    if not _db_ready:
        return default_state()
    try:
        with db_connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT value FROM runner_state WHERE state_key = 'runner_state'")
            row = cur.fetchone()
        return row["value"] if row and isinstance(row["value"], dict) else default_state()
    except Exception as e:
        LOG.error("durable state read failed: %s", e)
        return default_state()


def save_state(state: dict):
    if not _db_ready:
        raise RuntimeError("durable state unavailable; refusing ephemeral enforcement state")
    with _state_lock, db_connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO runner_state (state_key, value, updated_at) VALUES ('runner_state', %s, NOW()) "
            "ON CONFLICT (state_key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
            (json.dumps(state),),
        )


def trades_open(state) -> list:
    return [t for t in state["trades"] if not t.get("closed")]


def daily_trade_count(state) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state["daily"]["date"] != today:
        state["daily"] = {"date": today, "count": 0}
    return state["daily"]["count"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def hours_since(iso: str) -> float:
    try:
        then = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return 0.0
    return (utcnow() - then).total_seconds() / 3600.0


# ---------------------------------------------------------------- enforcer

def live_option_positions() -> dict:
    positions = api("GET", "/v2/positions")
    if isinstance(positions, dict):
        positions = positions.get("positions", [])
    out = {}
    for p in positions:
        sym = str(p.get("symbol", "")).upper()
        if p.get("class") == "us_option" or OCC_RE.match(sym):
            out[sym] = p
    return out


def recent_multileg_orders() -> list:
    """Read durable Alpaca order history to recover original spreads after restart."""
    try:
        orders = api("GET", "/v2/orders?status=all&nested=true&limit=500&direction=desc&after=2026-01-01T00:00:00Z")
        if isinstance(orders, dict):
            orders = orders.get("orders", [])
        return [o for o in orders if isinstance(o, dict) and o.get("order_class") == "mleg" and o.get("legs")]
    except CliError as e:
        LOG.error("cannot read order history for recovery: %s", e)
        return []


def recover_open_spreads(live: dict) -> list:
    """Map live legs to their original mleg order; preserve original submitted_at."""
    remaining = set(live)
    recovered = []
    for order in recent_multileg_orders():
        legs = [str(x.get("symbol", "")).upper() for x in order.get("legs", [])]
        matched = [s for s in legs if s in remaining]
        if len(matched) < 2:
            continue
        for s in matched:
            remaining.discard(s)
        root = re.match(r"^([A-Z]{1,6})", matched[0])
        symbol = root.group(1) if root else matched[0]
        recovered.append({
            "signal_id": f"RECOVERED-{order.get('id')}",
            "symbol": symbol,
            "opened_at": order.get("submitted_at") or utcnow().isoformat(),
            "legs": matched,
            "max_loss_usd": 600.0,
            "max_hold_hours": CFG["DEFAULT_MAX_HOLD_HOURS"],
            "order_id": order.get("id"),
            "closed": False,
            "close_pending": False,
            "recovered": True,
            "note": "recovered from original Alpaca mleg order history",
        })
    if remaining:
        # Fail closed: unknown legs are immediately due, never granted a fresh timer.
        recovered.append({
            "signal_id": f"RECOVERY_REVIEW-{utcnow().strftime('%H%M%S')}",
            "symbol": "/".join(sorted(remaining)),
            "opened_at": "1970-01-01T00:00:00+00:00",
            "legs": sorted(remaining),
            "max_loss_usd": 600.0,
            "max_hold_hours": 0.0,
            "order_id": None,
            "closed": False,
            "close_pending": False,
            "recovered": True,
            "note": "unmatched legs require immediate close/review; no fresh timer granted",
        })
    return recovered


def market_is_open() -> bool:
    try:
        clock = api("GET", "/v2/clock")
        return bool(clock.get("is_open"))
    except CliError as e:
        LOG.warning("clock check failed: %s", e)
        return False


def close_spread(t: dict, live: dict) -> dict:
    present = [s for s in t["legs"] if s in live]
    if len(present) < 2:
        raise CliError("spread close requires both legs still present")
    legs = []
    for sym in present:
        pos = live[sym]
        is_long = str(pos.get("side", "")).lower() == "long"
        qty = str(abs(int(float(pos.get("qty") or 1))))
        legs.append({
            "symbol": sym,
            "ratio_qty": qty,
            "side": "sell" if is_long else "buy",
            "position_intent": "sell_to_close" if is_long else "buy_to_close",
        })
    payload = {"order_class": "mleg", "qty": "1", "type": "market", "time_in_force": "day", "legs": legs}
    resp = api("POST", "/v2/orders", payload)
    return {"order_id": resp.get("id"), "response": resp}


def enforce_pass():
    state = load_state()
    live = live_option_positions()
    live_syms = set(live)
    tracked = trades_open(state)
    tracked_syms = {s for t in tracked for s in t["legs"]}
    changed = False

    # Recover only when durable state is missing/out of sync, from original Alpaca order history.
    orphan_syms = live_syms - tracked_syms
    if orphan_syms:
        recovered = recover_open_spreads({s: live[s] for s in orphan_syms})
        state["trades"].extend(recovered)
        tracked.extend(recovered)
        changed = True
        record_activity("recovered", "Recovered open option spread(s) from Alpaca order history", {"trades": recovered})
        LOG.warning("recovered %d spread(s) from order history", len(recovered))

    open_now = market_is_open()
    for t in tracked:
        present = [s for s in t["legs"] if s in live_syms]
        if not present:
            if not t.get("closed"):
                t["closed"] = True
                t["closed_at"] = utcnow().isoformat()
                t["close_reason"] = "reconcile: no live position found"
                changed = True
            continue
        net_pl = sum(float(live[s].get("unrealized_pl") or 0.0) for s in present)
        elapsed = hours_since(t["opened_at"])
        due_loss = net_pl <= -abs(float(t["max_loss_usd"]))
        due_hold = elapsed >= float(t["max_hold_hours"])
        if not (due_loss or due_hold):
            LOG.info("checked %s: net %+.2f, %.1fh/%.1fh, legs %d/%d — healthy", t["signal_id"], net_pl, elapsed, t["max_hold_hours"], len(present), len(t["legs"]))
            continue
        reason = "stop_loss" if due_loss else "hold_cap"
        if not t.get("due_reported"):
            record_activity(reason, f"{reason.upper()} DUE for {t['signal_id']}: net {net_pl:.2f}, {elapsed:.1f}h/{t['max_hold_hours']:.1f}h", {"signal_id": t["signal_id"], "legs": present, "net_pl": net_pl})
            t["due_reported"] = True
            changed = True
        if not open_now:
            continue
        if t.get("close_pending"):
            continue
        try:
            close = close_spread(t, live)
            t["close_pending"] = True
            t["close_order_id"] = close.get("order_id")
            t["close_submitted_at"] = utcnow().isoformat()
            changed = True
            record_activity("close_submitted", f"Submitted multi-leg close for {t['signal_id']}", {"signal_id": t["signal_id"], "order_id": t.get("close_order_id"), "legs": present})
        except Exception as e:
            record_activity("close_error", f"Close submission failed for {t['signal_id']}: {e}", {"signal_id": t["signal_id"], "legs": present})
            LOG.error("close failed for %s: %s", t["signal_id"], e)
    if changed:
        save_state(state)


# ---------------------------------------------------------------- scanning

def trading_enabled() -> bool:
    """Dashboard kill switch: trading is ON while the watchlist named
    KILLSWITCH_WATCHLIST exists on the account (toggled by the dashboard's
    on/off button via dashboard/api/control.js). Fail-closed on API errors.
    The enforcer is NOT gated — risk enforcement always runs."""
    if not CFG["REQUIRE_KILLSWITCH_ON"]:
        return True
    try:
        lists = api("GET", "/v2/watchlists")
        if isinstance(lists, dict):
            lists = lists.get("watchlists", [])
        return any(w.get("name") == CFG["KILLSWITCH_WATCHLIST"] for w in lists)
    except CliError as e:
        LOG.error("kill-switch check failed (%s) — treating as OFF (fail-closed)", e)
        record_activity("kill_switch_check_failed", f"kill-switch check failed: {e} — treating as OFF")
        return False


def market_is_open() -> bool:
    try:
        clock = api("GET", "/v2/clock")
        return bool(clock.get("is_open"))
    except CliError as e:
        LOG.warning("clock check failed (%s) — skipping scan pass", e)
        record_activity("clock_check_failed", f"Alpaca clock check failed: {e} — scan pass skipped")
        return False


def live_price(symbol: str) -> float:
    try:
        resp = api("GET", f"/v2/stocks/{symbol}/trades/latest")
        return float(resp.get("trade", {}).get("p") or 0.0)
    except (CliError, ValueError, AttributeError):
        return 0.0


def get_new_signals() -> list:
    src = CFG["SIGNAL_SOURCE"]
    if src == "off":
        return []
    if src == "simulated":
        spot = live_price("SPY") or 600.0
        return [{
            "signal_id": f"SIM-{utcnow().strftime('%Y%m%d-%H%M%S')}",
            "symbol": "SPY",
            "direction": "BUY",
            "underlying_price": spot,
            "fvg_context": {
                "gap_type": "bullish",
                "mss_confirmed": True,
                "displacement_strength": 1.2,
                "measured_move_target": round(spot + 15.0, 2),
                "entry_bar_timestamp": utcnow().isoformat(),
            },
            "note": "simulated signal for autonomy self-test",
        }]
    try:
        from signal_adapter import poll_signals
        return poll_signals(CFG["SYMBOLS"])
    except ImportError as e:
        LOG.error("signal_adapter import failed: %s — check live_bot.py deps "
                  "(pandas, numpy, pytz, alpaca-py) are installed", e)
        return []
    except Exception as e:
        LOG.error("signal adapter raised: %s", e)
        return []


def account_context(symbol: str, estimated_cost: float):
    from risk_guardian import AccountRiskContext
    acct = api("GET", "/v2/account")
    daily_pnl = float(acct.get("equity") or 0.0) - float(acct.get("last_equity") or 0.0)
    positions = api("GET", "/v2/positions")
    if isinstance(positions, dict):
        positions = positions.get("positions", [])
    exposure = 0.0
    for p in positions:
        s = str(p.get("symbol", "")).upper()
        if s == symbol or (s.startswith(symbol) and s[len(symbol):len(symbol) + 1].isdigit()):
            exposure += abs(float(p.get("market_value") or 0.0))
    # Count defined-risk spreads, not individual option legs.
    open_spread_count = len(trades_open(load_state()))
    return AccountRiskContext(
        current_daily_pnl_usd=round(daily_pnl, 2),
        open_position_count=open_spread_count,
        current_symbol_exposure_usd=round(exposure, 2),
        proposed_trade_cost_usd=estimated_cost,
    )


def extract_leg_symbols(obj) -> list:
    found = set()
    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "symbol" and isinstance(v, str) and OCC_RE.match(v.upper()):
                    found.add(v.upper())
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(obj)
    return sorted(found)


def find_order_id(executor_out: dict):
    """Defensively locate the Alpaca order id in the executor result."""
    if not isinstance(executor_out, dict):
        return None
    for key in ("order_id", "id"):
        if executor_out.get(key):
            return executor_out[key]
    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("id", "order_id") and isinstance(v, str) and len(v) >= 8:
                    return v
                r = walk(v)
                if r:
                    return r
        elif isinstance(node, list):
            for item in node:
                r = walk(item)
                if r:
                    return r
        return None
    return walk(executor_out.get("cli_response")) or walk(executor_out.get("order_payload"))


def scan_pass():
    if not trading_enabled():
        LOG.info("KILL SWITCH OFF (watchlist %s absent) — scanning paused; enforcer still active",
                 CFG["KILLSWITCH_WATCHLIST"])
        record_activity("kill_switch_off", "Kill switch OFF — scanning paused (enforcer still active)")
        return
    if not market_is_open():
        LOG.info("market closed — scan pass skipped")
        record_activity("market_closed", "Market closed — scan pass skipped")
        return
    signals = get_new_signals()
    if not signals:
        LOG.info("scan pass: no new signals")
        record_activity("no_signal", "Scan pass complete: no new FVG signals")
        return
    state = load_state()
    runs_dir = CFG["STATE_DIR"] / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    open_symbols = {t.get("symbol") for t in trades_open(state) if t.get("symbol")}

    for sig in signals:
        sid = sig.get("signal_id", "unknown")
        sym = sig.get("symbol", "")
        if sym in open_symbols:
            LOG.info("skipping %s: option trade already open on %s", sid, sym)
            record_activity("skipped_open", f"{sid}: skipped, {sym} already has an open tracked trade",
                             {"signal_id": sid, "symbol": sym})
            continue
        if daily_trade_count(state) >= CFG["MAX_DAILY_TRADES"]:
            LOG.warning("daily trade cap (%d) reached — signal %s skipped",
                        CFG["MAX_DAILY_TRADES"], sid)
            record_activity("daily_cap", "Daily trade cap (" + str(CFG["MAX_DAILY_TRADES"]) + f") reached — {sid} skipped",
                             {"signal_id": sid, "symbol": sym})
            break
        try:
            from pipeline import run_pipeline
            ctx = account_context(
                sym,
                float(sig.get("estimated_cost_usd") or CFG["ESTIMATED_TRADE_COST_USD"]),
            )
            result = run_pipeline(sig, ctx, dry_run=CFG["DRY_RUN"])
        except Exception as e:
            LOG.error("pipeline failed for %s: %s", sid, e)
            record_activity("pipeline_error", f"{sid}: pipeline raised {e}",
                             {"signal_id": sid, "symbol": sym, "source": sig.get("source")})
            (runs_dir / f"{utcnow().strftime('%Y%m%dT%H%M%S')}_{sid}_ERROR.json").write_text(
                json.dumps({"signal": sig, "error": str(e)}, indent=2, default=str))
            continue

        (runs_dir / f"{utcnow().strftime('%Y%m%dT%H%M%S')}_{sid}.json").write_text(
            json.dumps({"signal": sig, "result": result}, indent=2, default=str))

        rg = result.get("risk_guardian") or {}
        ex = result.get("executor") or {}
        sc = result.get("scout") or {}
        LOG.info("pipeline complete for %s: status=%s guardian_decision=%s executor_action=%s",
                 sid, result.get("final_status"),
                 rg.get("decision", "?"), ex.get("action", "?"))
        decision = rg.get("decision", "?")
        veto_reason = rg.get("veto_reason")
        record_activity(
            "veto" if "VETO" in str(decision).upper() else "decision",
            f"{sid} ({sym})[{sig.get('source') or 'unknown_source'}]: guardian={decision} executor="
            + str(ex.get("action", "?"))
            + (f" — {veto_reason}" if "VETO" in str(decision).upper() and veto_reason else ""),
            {"signal_id": sid, "symbol": sym, "source": sig.get("source"),
             "direction": sc.get("direction") or sig.get("direction"),
             "underlying_price": sc.get("underlying_price") or sig.get("underlying_price"),
             "thesis": sc.get("thesis"), "confidence": sc.get("confidence_score"),
             "strategy": sc.get("suggested_strategy"),
             "decision": decision, "veto_reason": veto_reason,
             "risk_rationale": rg.get("risk_rationale"),
             "max_loss_usd": rg.get("final_max_loss_usd"),
             "max_hold_hours": rg.get("final_max_hold_hours"),
             "executor_action": ex.get("action"), "executor_reason": ex.get("reason")},
        )

        if result.get("final_status") == "PROCESSED" and ex.get("action") == "ORDER_SUBMITTED":
            legs = extract_leg_symbols(ex)
            state["trades"].append({
                "signal_id": sid,
                "symbol": sym,
                "opened_at": utcnow().isoformat(),
                "legs": legs,
                "max_loss_usd": float(rg.get("final_max_loss_usd") or 600.0),
                "max_hold_hours": float(rg.get("final_max_hold_hours") or CFG["DEFAULT_MAX_HOLD_HOURS"]),
                "order_id": find_order_id(ex),
                "closed": False,
            })
            state["daily"]["count"] = daily_trade_count(state) + 1
            open_symbols.add(sym)
            LOG.info("ORDER SUBMITTED for %s — legs %s now under enforcement", sid, legs)
            record_activity("order_submitted", f"ORDER SUBMITTED for {sid}: legs {legs}",
                             {"signal_id": sid, "symbol": sym, "legs": legs, "order_id": find_order_id(ex),
                              "source": sig.get("source"),
                              "direction": sc.get("direction") or sig.get("direction"),
                              "underlying_price": sc.get("underlying_price") or sig.get("underlying_price"),
                              "thesis": sc.get("thesis"), "confidence": sc.get("confidence_score"),
                              "strategy": sc.get("suggested_strategy"), "decision": decision,
                              "max_loss_usd": rg.get("final_max_loss_usd"),
                              "max_hold_hours": rg.get("final_max_hold_hours")})
    save_state(state)


# ------------------------------------------------------------------- loops

def enforcer_loop():
    while not _stop.is_set():
        try:
            enforce_pass()
        except Exception as e:
            LOG.error("enforce pass failed: %s", e)
        _stop.wait(CFG["ENFORCE_INTERVAL_SEC"])


def scanner_loop():
    while not _stop.is_set():
        try:
            scan_pass()
        except Exception as e:
            LOG.error("scan pass failed: %s", e)
        _stop.wait(CFG["SCAN_INTERVAL_SEC"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single enforce + scan pass")
    ap.add_argument("--selftest", action="store_true", help="single pass, simulated signal, dry-run forced ON")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    CFG["STATE_DIR"].mkdir(parents=True, exist_ok=True)
    init_database()

    if args.selftest:
        CFG["SIGNAL_SOURCE"] = "simulated"
        CFG["DRY_RUN"] = True
        CFG["REQUIRE_KILLSWITCH_ON"] = False
        LOG.info("SELFTEST: dry-run forced ON, simulated signal, kill switch bypassed")
        enforce_pass()
        scan_pass()
        LOG.info("SELFTEST complete — check %s and %s", state_path(), CFG["STATE_DIR"] / "runs")
        return

    if args.once:
        enforce_pass()
        scan_pass()
        return

    LOG.info("autonomous runner v7 starting: enforce=%ds scan=%ds symbols=%s dry_run=%s "
             "signal_source=%s killswitch_watchlist=%s (required=%s) health_port=%d",
             CFG["ENFORCE_INTERVAL_SEC"], CFG["SCAN_INTERVAL_SEC"], CFG["SYMBOLS"],
             CFG["DRY_RUN"], CFG["SIGNAL_SOURCE"], CFG["KILLSWITCH_WATCHLIST"],
             CFG["REQUIRE_KILLSWITCH_ON"], CFG["HEALTH_PORT"])
    start_health_server()
    threads = [
        threading.Thread(target=enforcer_loop, name="enforcer", daemon=True),
        threading.Thread(target=scanner_loop, name="scanner", daemon=True),
    ]
    for t in threads:
        t.start()
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        _stop.set()
        LOG.info("shutting down")


if __name__ == "__main__":
    main()