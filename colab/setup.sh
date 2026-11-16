#!/bin/bash
# Usage: setup.sh <session>
# Prepares a new session, in this order, and is safe to run twice:
#   (a) rclone, at the pinned release, from its GitHub release page;
#   (b) the campaign's Drive key file, on the session's own disk with mode 600, nowhere else;
#   on a GPU session only, and after send.sh:
#   (c) the fast DeltaNet kernel package, in its own folder, so that a process can be started
#       with it importable or without it;
#   (d) the model, downloaded at one revision;
#   (e) the pins file: written when the sent commit has none, otherwise compared.
# It first checks that the image's transformers is the pinned version, and stops if it is not.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ $# -eq 1 ] || die "usage: setup.sh <session>"
session=$1
require_session "$session"

echo "== the image"
remote_python "$session" check_image.py 120 || die "the image check failed"
gpu=$(remote_value R_GPU)

echo "== (a) rclone"
remote_python "$session" install_rclone.py 300 || die "rclone was not installed"

echo "== (b) the Drive key file"
key=$(key_file) || die "cannot find the Windows profile folder; set CHECK2_KEY_FILE"
[ -f "$key" ] || die "no key file at $key"
remote_python "$session" place_key.py 60 "CHECK2_STAGE=prepare" || die "the key folder was not made"
incoming=$(remote_value R_INCOMING)
colab upload -s "$session" "$key" "$incoming" || die "the key file was not uploaded"
remote_python "$session" place_key.py 60 "CHECK2_STAGE=place" || die "the key file was not placed"

if [ "$gpu" != "1" ]; then
    echo "== (c), (d), (e) skipped: this session has no GPU"
    exit 0
fi

echo "== (c) the fast kernel package"
remote_python "$session" install_fast_kernel.py 300 || die "the fast kernel package was not installed"

echo "== (d) the model and (e) the pins"
remote_python "$session" model_and_pins.py 300 || die "the model or the pins step failed"
