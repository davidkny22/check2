#!/bin/bash
# Usage: send.sh <session> <commit>
# Packs the commit with git archive, uploads it, and unpacks it to /content/check-twice on the
# session in place of what was there. The commit's full hash becomes the one line of
# /content/check-twice/.sent-commit. Refuses a commit that does not exist, and refuses while a
# job is running on the session, since the job runs inside that folder.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ $# -eq 2 ] || die "usage: send.sh <session> <commit>"
session=$1
commit=$2
case "$commit" in
-*) die "'$commit' is not a commit" ;;
esac

gd=$(git_dir) || exit 1
full=$(git --git-dir="$gd" rev-parse --verify --quiet "$commit^{commit}") ||
    die "no commit '$commit' in $REPO_DIR"
require_session "$session"

archive=$(mktemp --suffix=.tar.gz) || die "cannot make a temporary file"
trap 'rm -f "$archive"' EXIT
git --git-dir="$gd" archive --format=tar.gz -o "$archive" "$full" || die "git archive failed"
bytes=$(stat -c %s "$archive")
echo "packed $full ($bytes bytes)"

remote_archive="/content/incoming-$full.tar.gz"
colab upload -s "$session" "$archive" "$remote_archive" || die "upload failed"
remote_python "$session" unpack_commit.py 300 \
    "CHECK2_ARCHIVE=$remote_archive" "CHECK2_ARCHIVE_BYTES=$bytes" "CHECK2_COMMIT=$full" ||
    die "unpacking on '$session' failed"
