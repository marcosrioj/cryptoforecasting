#!/bin/bash
# Activate virtual environment
source .venv/bin/activate

# Run cryptoforecast.py passing all arguments given to this script
EMAIL_USER=marcosrioj@gmail.com EMAIL_PASS=Trabalho@16 python3 watchlist_notify.py "$@"
