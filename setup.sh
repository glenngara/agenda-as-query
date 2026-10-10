#!/usr/bin/env bash
# One-time setup (macOS). Run: bash setup.sh
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo; echo "Setup done. Now open .env and paste your Anthropic and OpenAI keys."
