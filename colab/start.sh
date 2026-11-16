#!/bin/bash
# Usage: start.sh <session> <cpu|T4|L4|A100|G4> [--high-mem] [--minutes N]
# Creates a session under the given name. When the service refuses or does not answer, it
# asks again, at most once a minute, until N minutes (default 20) have passed; no attempt runs
# past that point. Ends by printing the status.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage="usage: start.sh <session> <cpu|T4|L4|A100|G4> [--high-mem] [--minutes N]"
[ $# -ge 2 ] || die "$usage"
session=$1
kind=$2
shift 2
high_mem=0
minutes=20
while [ $# -gt 0 ]; do
    case "$1" in
    --high-mem) high_mem=1 ;;
    --minutes)
        [ $# -ge 2 ] || die "$usage"
        minutes=$2
        shift
        ;;
    *) die "$usage" ;;
    esac
    shift
done
[[ "$minutes" =~ ^[0-9]+$ ]] || die "--minutes takes a whole number of minutes, not '$minutes'"
check_session_name "$session"

new_args=(new -s "$session")
case "$kind" in
cpu) ;;
T4 | L4 | A100 | G4) new_args+=(--gpu "$kind") ;;
*) die "unknown runtime '$kind'; use cpu, T4, L4, A100 or G4" ;;
esac
[ "$high_mem" -eq 1 ] && new_args+=(--high-mem)

if session_alive "$session"; then
    die "a session named '$session' is already alive"
fi

# The tool's request for a machine has no time limit of its own, and the service can leave it
# unanswered for good. Each request therefore gets a read limit, and the tool asks once more
# after one that timed out, first checking whether the machine was assigned after all. The
# whole attempt is cut off after ATTEMPT_SECONDS in any case, which the tool's two reads, its
# pause and its check fit inside.
export COLAB_CLI_REQUEST_TIMEOUT=40
export COLAB_CLI_ASSIGN_RETRIES=1
ATTEMPT_SECONDS=100
# Attempts start at most once a minute.
PERIOD_SECONDS=60
# An attempt shorter than this cannot complete, so none is started with less time left.
MIN_ATTEMPT_SECONDS=20

# Assignments the tool lists without a name ("[?]"): a request cut off after the service
# granted it leaves one, and it costs credits until it is stopped.
unnamed() {
    list_sessions | grep '^\[?\]' || true
}

before=$(unnamed)
deadline=$(($(date +%s) + minutes * 60))
attempt=0
while true; do
    attempt=$((attempt + 1))
    started=$(date +%s)
    limit=$((deadline - started))
    [ "$limit" -gt "$ATTEMPT_SECONDS" ] && limit=$ATTEMPT_SECONDS
    [ "$limit" -lt "$MIN_ATTEMPT_SECONDS" ] && limit=$MIN_ATTEMPT_SECONDS
    echo "attempt $attempt at $(date -u +%H:%M:%SZ), cut off after $limit s: colab ${new_args[*]}"
    out=$(timeout --kill-after=5 "$limit" "$COLAB_TOOL" --auth=oauth2 "${new_args[@]}")
    rc=$?
    printf '%s\n' "$out" | tail -n 3
    # A refusal can still leave an assignment behind, so the list of sessions decides.
    if session_alive "$session"; then
        echo "session '$session' created on attempt $attempt"
        break
    fi
    case "$rc" in
    124 | 137) echo "no answer within $limit s; the attempt was cut off" ;;
    *) echo "refused (tool exit $rc)" ;;
    esac
    after=$(unnamed)
    if [ "$after" != "$before" ]; then
        printf '%s\n' "$after"
        die "an assignment without a name appeared; it costs credits until it is stopped"
    fi
    next=$((started + PERIOD_SECONDS))
    now=$(date +%s)
    [ "$next" -lt "$now" ] && next=$now
    if [ $((deadline - next)) -lt "$MIN_ATTEMPT_SECONDS" ]; then
        die "no session after $attempt attempts in $minutes minutes"
    fi
    sleep $((next - now))
done

"$COLAB_DIR/status.sh"
