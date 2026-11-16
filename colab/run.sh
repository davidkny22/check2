#!/bin/bash
# Usage: run.sh <session> <job> [--builtin-path] -- <command...>
# Starts the command on the session in /content/check-twice, detached, and returns at once. Its
# output goes to /content/jobs/<job>/log.txt and its exit code to /content/jobs/<job>/exit_code
# when it ends. The process sees CHECK2_SESSION set to the session's name. With --builtin-path
# the fast DeltaNet kernel package cannot be imported in it. Refuses a job name that is running.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage="usage: run.sh <session> <job> [--builtin-path] -- <command...>"
[ $# -ge 2 ] || die "$usage"
session=$1
job=$2
shift 2
builtin_path=0
if [ "${1:-}" = "--builtin-path" ]; then
    builtin_path=1
    shift
fi
[ "${1:-}" = "--" ] || die "$usage"
shift
[ $# -ge 1 ] || die "$usage"
[[ "$job" =~ ^[A-Za-z0-9._-]+$ ]] && [[ "$job" != .* ]] ||
    die "job name '$job' may hold only letters, digits, '.', '-' and '_', and not start with '.'"
require_session "$session"

# The command travels as a JSON list, so no word of it is read by a shell on its way.
argv_json=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@") ||
    die "cannot encode the command"
remote_python "$session" start_job.py 120 \
    "CHECK2_JOB=$job" "CHECK2_ARGV=$argv_json" "CHECK2_SESSION_NAME=$session" \
    "CHECK2_BUILTIN_PATH=$builtin_path" ||
    die "the job was not started"
