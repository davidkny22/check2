#!/bin/bash
# Usage: drive.sh <session> <to|from> <session path> <drive path>
# Copies a file or a folder between the session and the Drive folder, through the remote
# check2drive: that the key file roots at that folder. A plain rclone copy: nothing is moved,
# synced or deleted. Afterwards every copied file must have the same size on both sides.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage="usage: drive.sh <session> <to|from> <session path> <drive path>"
[ $# -eq 4 ] || die "$usage"
session=$1
direction=$2
session_path=$3
drive_path=$4
case "$direction" in
to | from) ;;
*) die "$usage" ;;
esac
require_session "$session"

remote_python "$session" drive_copy.py 180 "CHECK2_DIRECTION=$direction" \
    "CHECK2_SESSION_PATH=$session_path" "CHECK2_DRIVE_PATH=$drive_path" ||
    die "the copy $direction Drive did not complete"
