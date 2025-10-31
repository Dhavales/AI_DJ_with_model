
  # DJ Mobile Web App

  This is a code bundle for DJ Mobile Web App. The original project is available at https://www.figma.com/design/uCmVU1Y5OoNTzau4DPvDPu/DJ-Mobile-Web-App.

  ## Running the code

  Run `npm i` to install the dependencies.

  Run `npm run dev` to start the development server.
  
  
  ### Backend wiring
  - The app can call a mix‑planning backend at `POST /mix/plan` (like the one in `Genie-AI-DJ/apps/backend`).
  - Configure the base URL via Vite env: create a `.env` file next to `package.json` with:
    
    ```env
    VITE_API_BASE=http://localhost:3001
    ```
  
  - Generate a plan: pick tracks in the two “Up Next” lists, then click “Generate Mix”. The app calls `${VITE_API_BASE}/mix/plan` and renders the returned steps.
  - If the backend has no OpenAI key, the route should still return a safe fallback plan.
  
  ### Notes
  - This bundle is intended to replace the Streamlit `dj_interface.py` for UI. Audio playback is currently a stub; planning results are visualized as a table of steps.
  - To wire library search or audio streaming, point the search bar and deck players to your endpoints (e.g., a local library API or Spotify).
