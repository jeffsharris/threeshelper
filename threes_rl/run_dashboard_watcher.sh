#!/bin/sh
set -eu

cd /Users/jeffharris/code/threeshelper
exec .venv/bin/python -m threes_rl.dashboard --watch --interval 10 --refresh-seconds 15
