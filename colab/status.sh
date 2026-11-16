#!/bin/bash
# Usage: status.sh
# Prints the live sessions and the account's compute-unit balance.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ $# -eq 0 ] || die "usage: status.sh"

echo "== sessions"
list_sessions || die "the tool could not list the sessions"
echo "== balance"
colab usage || die "the tool could not read the balance"
