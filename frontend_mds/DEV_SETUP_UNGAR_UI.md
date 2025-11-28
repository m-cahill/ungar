# RediAI Frontend + UNGAR UI Setup Guide

This guide explains how to bring up the RediAI frontend environment to verify UNGAR card-game workflows.

## 1. Start the RediAI Stack

You need the backend, registry, and frontend running.

```bash
# From the RediAI root directory
# (Adjust to your actual script/compose commands if different)
docker compose up -d registry backend frontend

# OR, if running manually:
# 1. Start Registry/Backend
python scripts/deploy_helm.py --environment development --install

# 2. Start Frontend
cd frontend
npm install
npm run dev
```

## 2. Point UNGAR to this RediAI Instance

(This part is usually done in the UNGAR repo, but listed here for context)

```bash
export REDIAI_BASE_URL=http://localhost:8000
```

## 3. Seed UNGAR Data

To verify the UI, you need at least one UNGAR workflow in the registry.

```bash
# Run a tiny UNGAR training run that logs overlays & reward decomposition
# (This assumes you have access to ungar_bridge or a seed script)
python -m ungar_bridge.rediai_high_card_demo
# OR
python -m ungar_bridge.rediai_spades_demo
```

If you don't have the UNGAR repo, you can use the mock seeder in RediAI (if available) or rely on the `UngarDemoPage` mock mode.

## 4. Verify in UI

1. Open http://localhost:5173 (or your frontend URL).
2. In development mode, you'll be automatically authenticated.
3. Navigate to **UNGAR Demo** from the sidebar menu.
4. You should see:
   - "UNGAR Demo" heading
   - 4×14 card overlay grid (suits × ranks)
   - Reward decomposition table with bar chart
   - Mock Mode toggle (enabled by default)

### Available Routes

- `/ungar` - UNGAR workspace landing page
- `/ungar/demo` - UNGAR Demo with overlay grid and reward table
- `/xai` - XAI Overlay Demo (FiLM network analysis)
- `/xai/episodes` - Multi-game XAI episode traces
- `/workflows` - Workflow Registry dashboard

## 5. Troubleshooting

### "No routes matched" Error

If you see blank pages or "No routes matched" in the console:

1. **Check for stale .js files:**
   ```bash
   cd frontend
   bash scripts/check_js_shadowing.sh
   ```
   If shadowing files are detected, delete them.

2. **Restart the dev server:**
   ```bash
   # Kill the current server (Ctrl+C)
   npm run dev
   ```

3. **Clear Vite cache:**
   ```bash
   rm -rf node_modules/.vite
   npm run dev
   ```

### Mock Mode vs Real Data

The UNGAR Demo page has a "Mock Mode" toggle:
- **Enabled (default):** Uses hardcoded demo data, works without backend
- **Disabled:** Attempts to fetch from RediAI Registry API (requires backend running)

For initial UI verification, Mock Mode is recommended.
