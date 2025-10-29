#!/bin/bash
# Activate virtual environment
source .venv/bin/activate

# Run cryptoforecast.py passing all arguments given to this script
python3 cryptoforecast.py "$@"
