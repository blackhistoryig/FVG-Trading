#!/usr/bin/env python3
"""
FVG Copilot — Autonomous Runner (v4: free-tier Render web service)
==================================================================
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
    open option position with default caps (ADOPT_* trade records). No
    position can ever be left unmanaged; worst case is a default cap
    instead of the Risk Guardian's custom one.
Everything else identical to v3 (kill switch, enforcer, scanner, reconcile).
============================================================================
"""
import argparse
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
}

OCC_RE = re.compile(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$")
_state_lock = threading.Lock()
_stop = threading.Event()

# ------------------------------------------------------------ health server

class HealthHandler(BaseHTTPRequestHandler):
    """Minimal 200 OK endpoint — keeps Render's free web service awake and
    gives UptimeRobot something to ping. Same pattern as live_bot.py."""

    def _respond(self):
        body = b"OK"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._respond()

    def do_HEAD(self):
        self._respond()

    def log_message(self, format, *args):
        return


def start_health_server():
    server = HTTPServer(("0.0.0.0", CFG["HEALTH_PORT"]), HealthHandler)
    LOG.info("health check server on port %d (Render keep-alive)", CFG["HEALTH_PORT"])
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

def state_path() -> Path:
    return CFG["STATE_DIR"] / "agent_state.json"


def load_state() -> dict:
    p = state_path()
    if not p.exists():
        return {"trades": [], "daily": {"date": "", "count": 0}}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        LOG.warning("state file unreadable — starting from live-account reconcile")
        return {"trades": [], "daily": {"date": "", "count": 0}}


def save_state(state: dict):
    with _state_lock:
        tmp = state_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(state_path())


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
        then = datetime.fromisoformat(iso)
    except ValueError:
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


def close_position(sym: str, pos: dict) -> dict:
    try:
        resp = api("DELETE", f"/v2/positions/{sym}")
        LOG.info("closed %s via position-close endpoint: %s", sym, json.dumps(resp)[:200])
        return {"closed": True, "method": "position_close", "response": resp}
    except CliError as e:
        LOG.warning("position-close failed for %s (%s) — falling back to order", sym, e)
    side = "sell_to_close" if pos.get("side") == "long" else "buy_to_close"
    qty = abs(int(float(pos.get("qty") or 1)))
    order = {"symbol": sym, "qty": qty, "side": side, "type": "market", "time_in_force": "day"}
    resp = api("POST", "/v2/orders", order)
    LOG.info("closed %s via %s order: %s", sym, side, json.dumps(resp)[:200])
    return {"closed": True, "method": "order_fallback", "response": resp}


def enforce_pass():
    state = load_state()
    live = live_option_positions()
    live_syms = set(live)
    tracked_syms = set()
    changed = False

    for t in trades_open(state):
        tracked_syms.update(t["legs"])
        present = [s for s in t["legs"] if s in live_syms]
        if not present:
            t["closed"] = True
            t["closed_at"] = utcnow().isoformat()
            t["close_reason"] = "reconcile: no live position found"
            changed = True
            LOG.info("reconciled %s -> CLOSED (not on account)", t["signal_id"])
            continue
        net_pl = sum(float(live[s].get("unrealized_pl") or 0.0) for s in present)
        elapsed = hours_since(t["opened_at"])
        if net_pl <= -abs(float(t["max_loss_usd"])):
            LOG.warning("STOP-LOSS HIT for %s: net %.2f <= -%.2f after %.1fh — closing",
                        t["signal_id"], net_pl, t["max_loss_usd"], elapsed)
            for s in present:
                close_position(s, live[s])
            t["closed"] = True
            t["closed_at"] = utcnow().isoformat()
            t["close_reason"] = f"max loss {t['max_loss_usd']} hit (net {net_pl:.2f})"
            changed = True
        elif elapsed >= float(t["max_hold_hours"]):
            LOG.warning("HOLD CAP HIT for %s: %.1fh >= %.1fh, net %.2f — closing",
                        t["signal_id"], elapsed, t["max_hold_hours"], net_pl)
            for s in present:
                close_position(s, live[s])
            t["closed"] = True
            t["closed_at"] = utcnow().isoformat()
            t["close_reason"] = f"max hold {t['max_hold_hours']}h exceeded ({elapsed:.1f}h)"
            changed = True
        else:
            LOG.info("checked %s: net %+.2f, %.1fh/%.1fh, legs %d/%d — healthy",
                     t["signal_id"], net_pl, elapsed, t["max_hold_hours"],
                     len(present), len(t["legs"]))

    orphans = live_syms - tracked_syms
    if orphans:
        if CFG["ADOPT_UNKNOWN_POSITIONS"]:
            state["trades"].append({
                "signal_id": f"ADOPTED-{utcnow().strftime('%H%M%S')}",
                "symbol": "/".join(sorted(orphans)),
                "opened_at": utcnow().isoformat(),
                "legs": sorted(orphans),
                "max_loss_usd": 600.0,
                "max_hold_hours": CFG["DEFAULT_MAX_HOLD_HOURS"],
                "order_id": None,
                "closed": False,
                "note": "adopted from live account; caps are env defaults",
            })
            changed = True
            LOG.warning("adopted untracked option positions with default caps: %s", sorted(orphans))
        else:
            LOG.error("UNTRACKED option positions on account (not adopting): %s", sorted(orphans))

    if changed:
        save_state(state)


# ---------------------------------------------------------------- scanning

def trading_enabled() -> bool:
    """Dashboard kill switch: trading is ON while the watchlist named
    KILLSWITCH_WATCHLIST exists on the account. Fail-closed on API errors.
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
        return False


def market_is_open() -> bool:
    try:
        clock = api("GET", "/v2/clock")
        return bool(clock.get("is_open"))
    except CliError as e:
        LOG.warning("clock check failed (%s) — skipping scan pass", e)
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
    return AccountRiskContext(
        current_daily_pnl_usd=round(daily_pnl, 2),
        open_position_count=len(positions),
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
        return
    if not market_is_open():
        LOG.info("market closed — scan pass skipped")
        return
    signals = get_new_signals()
    if not signals:
        LOG.info("scan pass: no new signals")
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
            continue
        if daily_trade_count(state) >= CFG["MAX_DAILY_TRADES"]:
            LOG.warning("daily trade cap (%d) reached — signal %s skipped",
                        CFG["MAX_DAILY_TRADES"], sid)
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
            (runs_dir / f"{utcnow().strftime('%Y%m%dT%H%M%S')}_{sid}_ERROR.json").write_text(
                json.dumps({"signal": sig, "error": str(e)}, indent=2, default=str))
            continue

        (runs_dir / f"{utcnow().strftime('%Y%m%dT%H%M%S')}_{sid}.json").write_text(
            json.dumps({"signal": sig, "result": result}, indent=2, default=str))

        rg = result.get("risk_guardian") or {}
        ex = result.get("executor") or {}
        LOG.info("pipeline complete for %s: status=%s guardian_decision=%s executor_action=%s",
                 sid, result.get("final_status"),
                 rg.get("decision", "?"), ex.get("action", "?"))

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

    LOG.info("autonomous runner v4 starting: enforce=%ds scan=%ds symbols=%s dry_run=%s "
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
