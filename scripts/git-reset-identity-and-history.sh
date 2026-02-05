#!/usr/bin/env bash
#
# Git: change username, email, and optionally reset/rewrite commit history.
# Usage:
#   ./scripts/git-reset-identity-and-history.sh [--global] [--rewrite-history | --wipe-history]
#
# Options:
#   --global          Update user.name/user.email globally (~/.gitconfig)
#   --rewrite-history Rewrite all commits to use the new name/email
#   --wipe-history    Remove all commit history (keep files, new initial commit)
#

set -e

GLOBAL=""
REWRITE_HISTORY=true
WIPE_HISTORY=true

for arg in "$@"; do
  case "$arg" in
    --global)        GLOBAL="--global" ;;
    --rewrite-history) REWRITE_HISTORY=true ;;
    --wipe-history)  WIPE_HISTORY=true ;;
    -h|--help)
      echo "Usage: $0 [--global] [--rewrite-history | --wipe-history]"
      echo ""
      echo "  --global          Set name/email in global config (default: this repo only)"
      echo "  --rewrite-history Rewrite all commits to use new name/email"
      echo "  --wipe-history    Delete all history and create a single new initial commit"
      exit 0
      ;;
  esac
done

echo "=== Git identity and history reset ==="
echo ""

# Current values (repo or global)
CURRENT_NAME=$(git config $GLOBAL user.name 2>/dev/null || echo "")
CURRENT_EMAIL=$(git config $GLOBAL user.email 2>/dev/null || echo "")

echo "Current user.name:  ${CURRENT_NAME:-<not set>}"
echo "Current user.email: ${CURRENT_EMAIL:-<not set>}"
echo ""

read -p "New user.name:  " NEW_NAME
read -p "New user.email: " NEW_EMAIL

if [[ -z "$NEW_NAME" || -z "$NEW_EMAIL" ]]; then
  echo "Error: user.name and user.email are required."
  exit 1
fi

git config $GLOBAL user.name  "$NEW_NAME"
git config $GLOBAL user.email "$NEW_EMAIL"
echo "Updated Git identity."
echo ""

if [[ "$REWRITE_HISTORY" == true ]]; then
  echo "Rewriting commit history with new author..."
  export FILTER_BRANCH_SQUELCH_WARNING=1
  git filter-branch -f --env-filter "
    export GIT_AUTHOR_NAME=\"$NEW_NAME\"
    export GIT_AUTHOR_EMAIL=\"$NEW_EMAIL\"
    export GIT_COMMITTER_NAME=\"$NEW_NAME\"
    export GIT_COMMITTER_EMAIL=\"$NEW_EMAIL\"
  " --tag-name-filter cat -- --all
  echo "History rewritten. Run 'git push --force' if you need to update the remote."
fi

if [[ "$WIPE_HISTORY" == true ]]; then
  echo "Wiping commit history (keeping all files)..."
  BRANCH=$(git branch --show-current)
  git checkout --orphan new_main
  git add -A
  git commit -m "Initial commit"
  git branch -D "$BRANCH"
  git branch -m "$BRANCH"
  echo "History wiped. Run 'git push --force' if you need to update the remote."
fi

echo ""
echo "Done. New identity: $NEW_NAME <$NEW_EMAIL>"
