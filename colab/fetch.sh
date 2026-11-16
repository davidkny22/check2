#!/bin/bash
# Usage: fetch.sh <session> <remote path> <local path>
# Brings a file or a folder from the session to this machine. A file is written at the local
# path; a folder's contents are written into the local path, which is created when missing.
# The local path may be a WSL path or a Windows path. Checks the number of files and their
# total size against the session's.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ $# -eq 3 ] || die "usage: fetch.sh <session> <remote path> <local path>"
session=$1
remote=$2
dest=$(local_path "$3")
require_session "$session"

remote_python "$session" fetch_stage.py 300 "CHECK2_STAGE=pack" "CHECK2_PATH=$remote" ||
    die "could not pack '$remote' on '$session'"
kind=$(remote_value R_KIND)
archive=$(remote_value R_ARCHIVE)
want_files=$(remote_value R_FILES)
want_bytes=$(remote_value R_BYTES)

work=$(mktemp -d) || die "cannot make a temporary folder"
trap 'rm -rf "$work"' EXIT
colab download -s "$session" "$archive" "$work/payload.tar.gz" || die "download failed"
remote_python "$session" fetch_stage.py 60 "CHECK2_STAGE=clean" "CHECK2_PATH=$archive" ||
    echo "warning: the archive $archive was left on the session"
mkdir "$work/x" && tar -xzf "$work/payload.tar.gz" -C "$work/x" || die "cannot unpack the archive"

got_files=$(find "$work/x/payload" -type f | wc -l)
got_bytes=$(find "$work/x/payload" -type f -printf '%s\n' | awk '{s += $1} END {print s + 0}')
if [ "$got_files" != "$want_files" ] || [ "$got_bytes" != "$want_bytes" ]; then
    die "size check failed: the session has $want_files files, $want_bytes bytes; received $got_files files, $got_bytes bytes"
fi

if [ "$kind" = "file" ]; then
    mkdir -p "$(dirname "$dest")" && cp "$work/x/payload" "$dest" || die "cannot write $dest"
else
    mkdir -p "$dest" && cp -r "$work/x/payload/." "$dest/" || die "cannot write into $dest"
fi
echo "fetched $kind '$remote' to $dest: $got_files files, $got_bytes bytes, sizes match"
