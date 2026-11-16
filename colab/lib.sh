# Shared definitions for the session scripts in this folder. Every script sources this file.

set -uo pipefail
# Windows reads the two output streams of a WSL command separately and can drop lines when
# both go to one place, so the scripts write everything to standard output, in order.
exec 2>&1

COLAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$COLAB_DIR")"
REMOTE_DIR="$COLAB_DIR/remote"
COLAB_TOOL="${COLAB_TOOL:-$HOME/.local/bin/colab}"

# The last line a helper prints when it reached its end. The tool exits with status 0 even when
# code on the session raised an error, so this line is the only proof that a helper finished.
DONE_MARK="remote step finished"

die() {
    echo "error: $*"
    exit 1
}

colab() {
    "$COLAB_TOOL" --auth=oauth2 "$@"
}

check_session_name() {
    [[ "$1" =~ ^[A-Za-z0-9_-]+$ ]] || die "session name '$1' may hold only letters, digits, '-' and '_'"
}

# Prints the tool's list of live sessions; fails when the tool cannot be reached.
list_sessions() {
    local out
    out=$(colab sessions) || {
        printf '%s\n' "$out"
        return 1
    }
    printf '%s\n' "$out"
}

session_alive() {
    local out
    out=$(list_sessions) || die "the tool could not list the sessions: $out"
    printf '%s\n' "$out" | grep -q "^\[$1\] "
}

require_session() {
    check_session_name "$1"
    session_alive "$1" || die "no live session named '$1'"
}

# remote_python SESSION HELPER IDLE_SECONDS [NAME=VALUE ...]
# Runs remote/HELPER on the session with remote/common.py in front of it, the NAME=VALUE pairs
# set in its environment. IDLE_SECONDS is how long the tool waits for the next line of output.
# Succeeds only when the helper printed DONE_MARK as its last line. The helper's output is
# shown as it arrives and kept in REMOTE_OUTPUT for remote_value.
remote_python() {
    local session=$1 helper=$2 idle=$3
    shift 3
    local code log rc last kv
    code=$(mktemp --suffix=.py) || die "cannot make a temporary file"
    log=$(mktemp) || die "cannot make a temporary file"
    cat "$REMOTE_DIR/common.py" "$REMOTE_DIR/$helper" >"$code" || die "cannot read $REMOTE_DIR/$helper"
    local env_args=()
    for kv in "$@"; do
        env_args+=(--env "$kv")
    done
    colab exec -s "$session" -f "$code" --timeout "$idle" "${env_args[@]}" | tee "$log"
    rc=${PIPESTATUS[0]}
    last=$(tr -d '\r' <"$log" | grep -v '^[[:space:]]*$' | tail -n 1)
    REMOTE_OUTPUT=$(tr -d '\r' <"$log")
    rm -f "$code" "$log"
    [ "$rc" -eq 0 ] && [ "$last" = "$DONE_MARK" ]
}

# remote_value NAME: the value a helper printed on a line "NAME=value" in its last run.
remote_value() {
    printf '%s\n' "${REMOTE_OUTPUT:-}" | sed -n "s/^$1=//p" | tail -n 1
}

# A path on this machine, given either as a WSL path or as a Windows path such as C:/x or C:\x.
local_path() {
    case "$1" in
    [A-Za-z]:[\\/]*) wslpath -u "$1" ;;
    *) printf '%s\n' "$1" ;;
    esac
}

# The campaign's Drive key file on this machine. CHECK2_KEY_FILE overrides the default, which is
# .config/check-twice/rclone.conf under the Windows user's profile folder.
key_file() {
    if [ -n "${CHECK2_KEY_FILE:-}" ]; then
        printf '%s\n' "$CHECK2_KEY_FILE"
        return 0
    fi
    local profile
    profile=$(cd /mnt/c && cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')
    [ -n "$profile" ] || return 1
    printf '%s/.config/check-twice/rclone.conf\n' "$(wslpath -u "$profile")"
}

# The git folder of the checkout these scripts sit in. A worktree's .git is a file that names
# its folder with a Windows path, which git inside WSL cannot follow by itself.
git_dir() {
    local dotgit="$REPO_DIR/.git" gd
    if [ -d "$dotgit" ]; then
        printf '%s\n' "$dotgit"
        return 0
    fi
    [ -f "$dotgit" ] || die "no git repository at $REPO_DIR"
    gd=$(sed -n 's/^gitdir: //p' "$dotgit" | tr -d '\r')
    gd=$(local_path "$gd")
    [ -d "$gd" ] || die "the git folder $gd named in $dotgit does not exist"
    printf '%s\n' "$gd"
}
