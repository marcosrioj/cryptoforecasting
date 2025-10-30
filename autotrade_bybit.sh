#!/bin/bash
# Activate virtual environment
source .venv/bin/activate

# Run cryptoforecast.py passing all arguments given to this script
export BYBIT_API_KEY=\"<your_test_api_key>\"
export BYBIT_API_SECRET=\"<your_test_api_secret>\"
python3 watchlist_notify.py --email-to marcosrioj@gmail.com
