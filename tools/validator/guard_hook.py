"""PreToolUse hook: deny Edit/Write/NotebookEdit on frozen validator paths, and Bash commands that name a
frozen path together with a mutating command. Reading, hashing and running the validator stay allowed.

Frozen paths: results/phase1/frozen.json "paths" (absolute or repo-relative path prefixes).
Exit 2 with a reason on stderr blocks the call.
"""
import json, os, re, sys

FROZEN = os.environ.get("VALIDATOR_FROZEN",
                        os.path.expanduser("~/nki-moe-megakernel/.claude/worktrees/q1q2/results/phase1/frozen.json"))
MUTATING = re.compile(r"(^|[\s;&|(])(rm|mv|cp|chmod|chown|chattr|truncate|tee|dd|install|ln|touch|unlink|rsync|"
                      r"sed\s+-i|perl\s+-[a-z]*i|patch|git\s+(checkout|restore|rm|mv|reset|stash))\b|>|open\([^)]*['\"][wa]")


def frozen_paths():
    try:
        return json.load(open(FROZEN))["paths"]
    except Exception:
        return []


def names(text, paths):
    t = text.replace("~", os.path.expanduser("~"))
    hits = []
    for p in paths:
        base = os.path.basename(p.rstrip("/"))
        if p in t or (base and re.search(rf"(^|[/\s'\"]){re.escape(base)}($|[\s'\"/])", t)):
            hits.append(p)
    return hits


def main():
    ev = json.load(sys.stdin)
    tool, inp = ev.get("tool_name", ""), ev.get("tool_input", {})
    paths = frozen_paths()
    if not paths:
        return 0
    if tool in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        f = os.path.realpath(inp.get("file_path") or inp.get("notebook_path") or "")
        for p in paths:
            if f == os.path.realpath(p) or f.startswith(os.path.realpath(p).rstrip("/") + "/"):
                print(f"Blocked: {f} is a frozen validator path (results/phase1/FROZEN.md).", file=sys.stderr)
                return 2
    if tool == "Bash":
        cmd = inp.get("command", "")
        hit = names(cmd, paths)
        if hit and MUTATING.search(cmd):
            print(f"Blocked: command may modify frozen validator path(s) {hit} (results/phase1/FROZEN.md).",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
