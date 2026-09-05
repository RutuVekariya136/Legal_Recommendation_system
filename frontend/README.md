# Frontend

React 18 + Vite 6 + Tailwind v4 + React Router 6 + Axios + Recharts.

## Run

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api -> http://localhost:8000)
```

The backend must be running (`uvicorn app.main:app --app-dir backend` from the repo root).
Override the API target with `VITE_API_TARGET=http://host:port npm run dev`.

## Build

```bash
npm run build        # -> dist/
npm run preview
```

## Layout

```
src/
  services/api.js       axios client (one function per endpoint)
  hooks/useAsync.js     fetch-on-deps hook
  components/           Layout, ui primitives (ScoreBar, Badge…),
                        RecommendationList, ExplanationPanel (Recharts breakdown)
  pages/                Home, Search, Case, Recommend, Compare, Upload, About
```
