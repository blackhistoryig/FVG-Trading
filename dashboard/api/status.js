export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  const keyId = process.env.ALPACA_API_KEY;
  const secret = process.env.ALPACA_SECRET_KEY;
  if (!keyId || !secret) {
    return res.status(200).json({ ok: false, error: "env_not_set" });
  }
  const base = "https://paper-api.alpaca.markets";
  const headers = {
    "APCA-API-KEY-ID": keyId,
    "APCA-API-SECRET-KEY": secret,
  };
  const RUNNER_ACTIVITY_URL =
    process.env.RUNNER_ACTIVITY_URL || "https://fvg-copilot-runner.onrender.com/activity";

  try {
    const [a, p, o, act] = await Promise.all([
      fetch(base + "/v2/account", { headers }),
      fetch(base + "/v2/positions", { headers }),
      fetch(base + "/v2/orders?status=all&limit=20&direction=desc", { headers }),
      // Best-effort: the Render runner's own /activity feed (no signal / VETO /
      // kill-switch-off / order-submitted / stop-loss / hold-cap events). If the
      // free-tier service is asleep or slow, this must never block the tiles.
      fetch(RUNNER_ACTIVITY_URL, { signal: AbortSignal.timeout(4000) }).catch(() => null),
    ]);
    if (!a.ok) throw new Error("account_http_" + a.status);
    const account = await a.json();
    const positions = p.ok ? await p.json() : [];
    const orders = o.ok ? await o.json() : [];

    let activity = { ok: false, events: [], error: "runner_unreachable" };
    if (act && act.ok) {
      try {
        activity = await act.json();
      } catch (e) {
        activity = { ok: false, events: [], error: String(e) };
      }
    }

    return res.status(200).json({
      ok: true,
      fetched_at: new Date().toISOString(),
      account: {
        portfolio_value: account.portfolio_value,
        cash: account.cash,
        equity: account.equity,
        unrealized_pl: account.unrealized_pl,
        last_equity: account.last_equity,
        status: account.status,
      },
      positions: positions.map((x) => ({
        symbol: x.symbol,
        qty: x.qty,
        side: x.side,
        asset_class: x.asset_class,
        avg_entry_price: x.avg_entry_price,
        current_price: x.current_price,
        unrealized_pl: x.unrealized_pl,
        market_value: x.market_value,
      })),
      orders: orders.map((x) => ({
        id: x.id,
        submitted_at: x.submitted_at,
        symbol: x.symbol,
        side: x.side,
        qty: x.qty,
        filled_qty: x.filled_qty,
        status: x.status,
        filled_avg_price: x.filled_avg_price,
        order_class: x.order_class,
        type: x.type,
        asset_class: x.asset_class,
        legs: (x.legs || []).map((l) => ({
          symbol: l.symbol,
          side: l.side,
          qty: l.qty,
          filled_avg_price: l.filled_avg_price,
          status: l.status,
        })),
      })),
      // Runner activity feed: what the autonomous bot is ACTUALLY doing right
      // now (no-signal scans, kill-switch state, VETOs, fills, stop-loss/hold-
      // cap closes). Comes straight from the Render service's in-memory feed.
      runner: {
        reachable: !!(act && act.ok),
        killswitch_watchlist: activity.killswitch_watchlist || null,
        require_killswitch_on: activity.require_killswitch_on ?? null,
        scan_interval_sec: activity.scan_interval_sec || null,
        events: activity.events || [],
      },
    });
  } catch (e) {
    return res
      .status(200)
      .json({ ok: false, error: String(e && e.message ? e.message : e) });
  }
}
