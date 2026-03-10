#!/bin/bash
# Auto-commit and push script before clear command

set -e

cd "D:\graduation_thesis"

# Check if there are any changes
if git diff --quiet && git diff --staged --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "No changes to commit. Skipping git operations."
    exit 0
fi

# Stage all changes (new, modified, deleted files)
git add -A

# Generate commit message based on changes
ADDED=$(git diff --cached --numstat | awk '$1=="-" {count++} END {print count+0}')
MODIFIED=$(git diff --cached --numstat | awk '$1!="-" && $2!="-" {count++} END {print count+0}')
DELETED=$(git diff --cached --numstat | awk '$2=="-" {count++} END {print count+0}')

MSG_PARTS=()
[ $ADDED -gt 0 ] && MSG_PARTS+=("added $ADDED file(s)")
[ $MODIFIED -gt 0 ] && MSG_PARTS+=("modified $MODIFIED file(s)")
[ $DELETED -gt 0 ] && MSG_PARTS+=("deleted $DELETED file(s)")

COMMIT_MSG="Auto-commit before clear: ${MSG_PARTS[*]}"

# Commit
git commit -m "$COMMIT_MSG"

# Push to remote
git push origin master

echo "Successfully committed and pushed changes."