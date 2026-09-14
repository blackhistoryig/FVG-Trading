// dashboard/reasoning.js — renders the live Agent Reasoning panel from the
// runner's /activity feed (v6 events: decision / veto / order_submitted with
// Scout thesis, confidence, strategy, source label, Guardian caps).
// Loaded after index.html's inline script (defer); reuses its global esc()
// helper and wraps window.render so every /api/status poll refreshes this
// panel alongside the tiles.
(function () {
  var escFn = (typeof window.esc === 'function') ? window.esc : function (s) { return String(s == null ? '' : s); };

  function renderReasoning(runner) {
    var termEl = document.getElementById('reasoning-terminal');
    var tb = document.getElementById('reasoning-body');
    if (!tb) return;
    var events = (runner && runner.events) || [];
    var dec = events.filter(function (e) {
      return e.kind === 'decision' || e.kind === 'veto' || e.kind === 'order_submitted';
    });
    if (!dec.length) {
      tb.innerHTML = '<tr><td colspan="5">No agent decisions yet this session — rows appear here live the moment Scout → Risk Guardian run.</td></tr>';
      if (termEl) termEl.innerHTML = '<div><span class="mut">waiting for the next signal&hellip;</span></div>';
      return;
    }
    tb.innerHTML = dec.slice(0, 12).map(function (e) {
      var x = e.extra || {};
      var t = e.ts ? new Date(e.ts).toLocaleString() : '—';
      var sig = (x.symbol || '?') + ' ' + (x.direction || '');
      if (x.underlying_price != null) sig += ' @ $' + x.underlying_price;
      if (x.source) sig += ' [' + x.source + ']';
      var scout = (x.strategy || '—');
      if (x.confidence != null) scout += ', conf ' + x.confidence;
      if (x.thesis) scout += '. ' + x.thesis;
      var g = x.decision || '?';
      var gtxt = g;
      if (x.veto_reason) gtxt += ' — ' + x.veto_reason;
      else if (x.risk_rationale) gtxt += ' — ' + x.risk_rationale;
      if (x.max_loss_usd != null) gtxt += ' (cap $' + x.max_loss_usd + ' / ' + x.max_hold_hours + 'h)';
      var ex = x.executor_action || '?';
      if (x.executor_reason && String(ex).indexOf('ORDER_SUBMITTED') < 0) ex += ' — ' + x.executor_reason;
      var cls = String(g).toUpperCase().indexOf('VETO') >= 0 ? 'veto' : (String(ex).indexOf('ORDER_SUBMITTED') >= 0 ? 'approve' : 'open');
      return '<tr><td>' + escFn(t) + '</td><td>' + escFn(sig) + '</td><td>' + escFn(scout) + '</td><td><span class="tag ' + cls + '">' + escFn(gtxt) + '</span></td><td>' + escFn(ex) + '</td></tr>';
    }).join('');
    if (termEl) {
      termEl.innerHTML = dec.slice(0, 6).map(function (e) {
        var x = e.extra || {};
        var t = e.ts ? new Date(e.ts).toLocaleTimeString() + ' ' : '';
        var src = x.source ? ' [' + x.source + ']' : '';
        var head = t + (x.symbol || '?') + ' ' + (x.direction || '') + src;
        var tagcls = e.kind === 'veto' ? 'err' : 'ok';
        var line1 = '<div style="margin-top:10px"><span class="mut">></span> <span class="cmd">' + escFn(head) + '</span><br>';
        var line2 = '<span class="mut">thesis:</span> ' + escFn(x.thesis || '—') + ' <span class="mut">&middot; confidence:</span> <span class="ok">' + escFn(x.confidence != null ? x.confidence : '—') + '</span><br>';
        var line3 = '<span class="' + tagcls + '">[' + escFn(x.decision || '?') + ']</span> <span class="mut">' + escFn(x.veto_reason || x.risk_rationale || '') + '</span> <span class="mut">&middot; executor: ' + escFn(x.executor_action || '?') + '</span></div>';
        return line1 + line2 + line3;
      }).join('');
    }
  }

  var origRender = window.render;
  if (typeof origRender === 'function') {
    window.render = function (d) {
      origRender(d);
      renderReasoning(d && d.runner);
    };
  }
  if (typeof window.loadStatus === 'function') window.loadStatus();
})();
