#!/bin/bash
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/activate" ]; then
    echo ""
    echo "  ERROR: Virtual environment not found."
    echo "  Please run ./install-linux.sh first."
    echo ""
    exit 1
fi

source .venv/bin/activate
streamlit run app.py
