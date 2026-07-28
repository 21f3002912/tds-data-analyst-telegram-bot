#!/usr/bin/env bash
set -e

python bot.py &
BOT_PID=$!

python webapp.py &
WEB_PID=$!

trap 'kill $BOT_PID $WEB_PID 2>/dev/null || true' EXIT INT TERM

wait -n $BOT_PID $WEB_PID