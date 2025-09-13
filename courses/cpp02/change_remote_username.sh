ROOT="$(pwd)"

find "$ROOT" -type d -name .git |
while IFS= read -r gitdir; do
  repo_dir="${gitdir%/.git}"

  cd "$repo_dir" || { echo "Failed to cd into $repo_dir"; continue; }

  old_url="$(git remote get-url origin)"

  if [[ "$old_url" == *VladislavBalabaev* ]]; then
    new_url="${old_url/VladislavBalabaev/vbalab}"
    git remote set-url origin "$new_url"
    echo "Updated '$repo_dir' → $new_url"
  fi
  cd "$ROOT"
done
