#!/bin/bash
# Activate virtual environment
source .venv/bin/activate

# Run cryptoforecast.py passing all arguments given to this script
export EMAIL_USER="marcosrioj@gmail.com"
export EMAIL_PASS="rwwx vduj cdxx xiar"
python3 watchlist_notify.py "$@"
