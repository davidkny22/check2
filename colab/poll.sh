#!/bin/bash
# Usage: poll.sh <session> <job> [--lines N]
# Prints "running" or "done <exit code>", the seconds since the job started, and the last N
# lines of its log (default 20). A job whose process vanished without an exit code is "lost",
# and the script then exits non-zero.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage="usage: poll.sh <session> <job> [--lines N]"
[ $# -eq 2 ] || [ $# -eq 4 ] || die "$usage"
session=$1
job=$2
lines=20
if [ $# -eq 4 ]; then
    [ "$3" = "--lines" ] || die "$usage"
    lines=$4
fi
[[ "$lines" =~ ^[0-9]+$ ]] || die "--lines takes a whole number, not '$lines'"
[[ "$job" =~ ^[A-Za-z0-9._-]+$ ]] && [[ "$job" != .* ]] || die "no job can be named '$job'"
require_session "$session"

remote_python "$session" job_report.py 60 "CHECK2_JOB=$job" "CHECK2_LINES=$lines" ||
    die "could not report on job '$job'"
