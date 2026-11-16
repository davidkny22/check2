#!/bin/bash
# Usage: stop.sh <session>
# Stops the session and releases its machine. Everything on its disk is lost. Ends by printing
# the status.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ $# -eq 1 ] || die "usage: stop.sh <session>"
session=$1
require_session "$session"

colab stop -s "$session" || die "the tool could not stop '$session'"
if session_alive "$session"; then
    die "'$session' is still listed after the stop"
fi
echo "stopped '$session' at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$COLAB_DIR/status.sh"
