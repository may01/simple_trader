#!/bin/bash
# Create Docker volumes for simple_trader.
set -e

if [ -n "$VOL_PATH" ]; then
    mkdir -p "$VOL_PATH/simple_trader_vol" "$VOL_PATH/simple_trader_vol_long"
    docker volume create --driver local \
        --opt type=none --opt o=bind \
        --opt device="$VOL_PATH/simple_trader_vol" \
        simple_trader_vol
    docker volume create --driver local \
        --opt type=none --opt o=bind \
        --opt device="$VOL_PATH/simple_trader_vol_long" \
        simple_trader_vol_long
else
    docker volume create simple_trader_vol
    docker volume create simple_trader_vol_long
fi

echo "Volumes created:"
docker volume ls --filter name=simple_trader
