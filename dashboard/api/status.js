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
  try {
    const [a, p, o] = await Promise.all([
      fetch(base + "/v2/account", { headers }),
      fetch(base + "/v2/positions", { headers }),
      fetch(base + "/v2/orders?status=all&limit=20&direction=desc", { headers }),
    ]);
    if (!a.ok) throw new Error("account_http_" + a.status);
    const account = await a.json();
    const positions = p.ok ? await p.json() : [];
    const orders = o.ok ? await o.json() : [];
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
    });
  } catch (e) {
    return res
      .status(200)
      .json({ ok: false, error: String(e && e.message ? e.message : e) });
  }
}
