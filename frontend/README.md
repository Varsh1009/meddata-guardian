# MedGuard AI – React frontend

Same functionality as the Streamlit app, with a modern UI and **no sidebar**. All steps and tabs are in the main content area.

## Run (development)

1. **Start the API** (from project root):
   ```bash
   cd d:\meddata-guardian
   pip install -r backend/requirements.txt   # if not already
   uvicorn backend.main:app --reload --app-dir .
   ```
   API: http://localhost:8000

2. **Start the React app** (run from the **frontend** folder):
   ```bash
   cd d:\meddata-guardian\frontend
   npm install
   npm run dev
   ```
   App: http://localhost:5173 (proxies `/api` to the backend)

## Features (same as Streamlit)

- **Step 1:** Onboarding (project description, model type, use case, timeline, data collection, location)
- **Step 2:** Load dataset (demo dropdown or CSV upload) → PHI scan
- **Step 3:** Generate synthetic twin
- **Step 4:** Analysis tabs:
  - **Data Quality** – missing values, duplicates, outliers
  - **Bias** – demographic distribution charts
  - **Deployment** – placeholder (full roadmap requires Ollama)
  - **Ask AI** – placeholder (Medical Advisor / Fairness Specialist require Ollama)
  - **Implement Changes** – select recommendations, apply, download modified CSV

## Build

```bash
cd frontend
npm run build
```

Static output in `frontend/dist`. Serve with any static host; set `VITE_API_URL` to your API base URL if not using the same origin.
