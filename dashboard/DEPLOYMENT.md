# Dashboard deployment

- Repo: blackhistoryig/FVG-Trading
- Production branch: hackathon/fvg-copilot
- Vercel root directory: dashboard
- Live URLs: fvg-copilot-status.vercel.app / fvg-copilot-status-poly-dash.vercel.app
- Files: index.html, reasoning.js, api/status.js, api/control.js
- Env vars (set in Vercel): ALPACA_API_KEY, ALPACA_SECRET_KEY
- The runner activity feed is fetched from https://fvg-copilot-runner.onrender.com/activity (override with RUNNER_ACTIVITY_URL)
