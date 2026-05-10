# Deploying the demo to Streamlit Community Cloud

This is the step-by-step for putting the Snapdragon Yield Analytics demo
behind a public URL so a recruiter can click and use it. The target
platform is Streamlit Community Cloud (`share.streamlit.io`) because it
is free, links directly to a public GitHub repo, gives a stable HTTPS
URL, redeploys on every push to `main`, and exposes a live log viewer
on the manage page.

## Prerequisites

- The GitHub repo holding this project is public.
- An Anthropic API key with credits, copied from
  [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).
- A Streamlit Community Cloud account, signed in with the same GitHub
  identity that owns the repo.

## Files that matter for the deploy

- `requirements.txt`  --  Streamlit Cloud installs from this on first
  boot and on each redeploy. Already present.
- `data/chip_production.db`  --  the synthetic SQLite database, ~12 MB,
  committed to the repo so the deployed container has data on first
  boot. Deterministic from `seed=42`; regenerate any time with
  `python data/generate_data.py && python data/setup_database.py`.
- `ui/app.py`  --  the Streamlit entry point. Streamlit Cloud auto-detects
  this if the file path is set during app creation.
- `agent/logging_setup.py`  --  attaches a stderr handler so every tool
  call appears in the Streamlit Cloud "manage app" log viewer in real
  time, alongside the per-run file at `logs/agent_*.log` (which is
  ephemeral on the hosted container).

## Steps

1. **Push the project to GitHub** as a public repo. Confirm
   `data/chip_production.db` is committed (it was previously gitignored
   and is now eligible to be tracked).

2. **Open `share.streamlit.io`** and click "Create app". Pick "Deploy a
   public app from GitHub", select the repo, set the branch to `main`,
   and set the main file path to `ui/app.py`.

3. **Set the Anthropic key as a secret.** In the app's "Advanced
   settings -> Secrets" field, paste:

       ANTHROPIC_API_KEY = "sk-ant-..."

   Streamlit injects secrets as environment variables, so the existing
   `os.getenv("ANTHROPIC_API_KEY")` lookup in `agent/agent.py` picks it
   up without code changes. Do not paste the literal key into the
   repo's `.env`; that file is gitignored for a reason.

4. **Deploy.** First boot takes 1-2 minutes because pip installs the
   dependency tree. Subsequent redeploys after a push to `main` take
   about 20-30 seconds.

5. **Verify.** Once the app loads, run "How is yield today?" from the
   sample-question buttons. On the "Manage app" page (the three-dot
   menu next to the live URL), open the log viewer and confirm you see
   lines like:

       2026-05-09 18:42:11 INFO snapdragon_agent.agent run start question='How is yield today?' ...
       2026-05-09 18:42:12 INFO snapdragon_agent.tools tool=query_database status=ok duration_ms=42 args={...}

   That is the same content as the local `logs/agent_*.log` file, plus
   anything Streamlit's own runtime emits.

## Things to know

- **The log file inside the container is ephemeral.** It is recreated
  on every container restart (which happens on redeploys, on inactivity
  sleeps, and occasionally on platform maintenance). The stderr stream
  is the durable view; it is captured by Streamlit Cloud and
  searchable on the manage page.

- **Inactivity sleep.** Streamlit Community Cloud puts inactive free
  apps to sleep. The first hit after a sleep takes about 30 seconds to
  cold-start. Acceptable for a portfolio demo; if a recruiter is
  expected at a specific time, click the URL once a few minutes
  beforehand to wake it up.

- **API cost.** Every recruiter session burns Anthropic credits on the
  configured key. With prompt caching the cost per question lands in
  the low single-digit cents on Sonnet 4.6 (a typical four-iteration
  yield-drop investigation runs on roughly 13 K cached input tokens,
  3 K fresh input tokens, and 1.5 K output tokens). Console billing
  has alerts available; configure one if peace of mind matters.

- **Public URL.** The app URL is indexable by anyone. If discoverability
  is a concern, gate it behind a simple shared password using
  `st.secrets["DEMO_PASSWORD"]`; a five-line check at the top of
  `ui/app.py` is enough.

- **Regenerating the database.** If the data generator changes, rebuild
  the DB locally and commit the new `data/chip_production.db`. The
  hosted app picks it up on the next push to `main`.

## Local sanity check before pushing

Run these once before you push the deploy commit:

    ./venv/bin/pytest -q
    streamlit run ui/app.py

Open the local URL, click a sample question, and confirm the trace
panel populates and the log lines stream into the terminal. If both
work locally, the hosted deploy will work too.
