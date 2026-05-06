#!/usr/bin/env bash
# One-command setup for the Snapdragon Yield Analytics demo.
# Creates a venv, installs dependencies, generates the synthetic dataset,
# and loads it into SQLite. Run from the project root.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Creating virtual environment in ./venv"
"$PYTHON_BIN" -m venv venv

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Upgrading pip"
pip install --quiet --upgrade pip

echo "==> Installing dependencies"
pip install --quiet -r requirements.txt

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example (remember to add your Anthropic API key)"
  cp .env.example .env
fi

echo "==> Generating synthetic dataset"
python data/generate_data.py

echo "==> Loading dataset into SQLite"
python data/setup_database.py

echo
echo "Setup complete."
echo "Next steps:"
echo "  1. Edit .env and set ANTHROPIC_API_KEY"
echo "  2. Run: streamlit run ui/app.py"
