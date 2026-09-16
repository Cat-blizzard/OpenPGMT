#!/usr/bin/env bash
# Read-only helpers shared by the PP4 package checker and installer.

isaacgym_python_versions() {
  local listing
  if ! listing=$(tar --force-local -tf "$1"); then
    echo "[!] Cannot read Isaac Gym archive: $1" >&2
    return 1
  fi
  printf '%s\n' "$listing" \
    | sed -nE 's@.*(^|/)gym_3([0-9]+)\.so$@\2@p' \
    | sort -nu | sed 's/^/3./'
}

isaacgym_select_python() {
  local versions selected requested="$2"
  versions=$(isaacgym_python_versions "$1") || return 1
  if [[ -z "$versions" ]]; then
    echo "[!] No Linux gym_3X.so Python bindings found in archive" >&2
    return 1
  fi
  if [[ -n "$requested" ]]; then
    if ! grep -Fx "$requested" <<< "$versions" >/dev/null; then
      echo "[!] Python $requested has no binding in this archive (available: $versions)" >&2
      return 1
    fi
    selected="$requested"
  else
    selected=$(printf '%s\n' "$versions" | tail -n 1)
  fi
  case "$selected" in
    3.8|3.9|3.10|3.11) printf '%s\n' "$selected" ;;
    *) echo "[!] Python $selected is outside this project's torch 2.1.2 configuration" >&2; return 1 ;;
  esac
}
