"""Claude Code PreToolUse hook: never let an agent trigger the Nordic AI Cup
EVALUATION attempt (one attempt per team, irreversible).

The harness runs this before Bash, PowerShell, WebFetch, Agent and Workflow
calls and feeds the tool input as JSON on stdin. If the input references the
evaluation endpoint, the call is denied. Validation, verify and status calls
are not affected, and neither is the local evaluator script.

Blocked when the tool input (lowercased) contains:
  - "evaluate/queue"                                       (the endpoint path)
  - "nordicaicup" together with "/evaluate", "evaluate/", "evaluate_",
    "queue evaluation", "queue_evaluation", "queueevaluation" or
    "evaluation attempt"                                  (any wording of it)

Unlock: the team lead creates the file .claude/EVAL_UNLOCK in the repo root
with a line "unlock-after: 2026-09-20T12:00" (local time). Calls are allowed
only after that moment. The agent must never create or edit that file; the
file is gitignored so it never travels with the repo.

Every blocked attempt is appended to .claude/eval-block.log with a timestamp.
"""
import datetime
import json
import os
import re
import sys

PATTERNS = [
    re.compile(r'evaluate/queue'),
    re.compile(r'nordicaicup.*(/evaluate|evaluate/|evaluate_|queue[ _]?evaluation|evaluation attempt)', re.S),
    re.compile(r'(/evaluate|evaluate/|evaluate_|queue[ _]?evaluation|evaluation attempt).*nordicaicup', re.S),
]


def project_dir() -> str:
    return os.environ.get('CLAUDE_PROJECT_DIR') or os.getcwd()


def unlocked() -> bool:
    path = os.path.join(project_dir(), '.claude', 'EVAL_UNLOCK')
    try:
        for line in open(path, encoding='utf-8'):
            m = re.match(r'\s*unlock-after:\s*(\S+)', line)
            if m:
                when = datetime.datetime.fromisoformat(m.group(1))
                return datetime.datetime.now() >= when
    except FileNotFoundError:
        return False
    except Exception:
        return False
    return False


def main() -> int:
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {'raw': raw}
    text = json.dumps(data.get('tool_input', data), ensure_ascii=False).lower()
    if not any(p.search(text) for p in PATTERNS):
        return 0
    if unlocked():
        return 0
    try:
        with open(os.path.join(project_dir(), '.claude', 'eval-block.log'), 'a', encoding='utf-8') as f:
            f.write(f'{datetime.datetime.now().isoformat(timespec="seconds")} blocked {data.get("tool_name", "?")}: {text[:300]}\n')
    except Exception:
        pass
    reason = ('BLOCKED by .claude/hooks/block-evaluation.py: this call references the Nordic AI Cup '
              'EVALUATION endpoint. The team has one evaluation attempt; only a human queues it. '
              'To allow, the team lead writes "unlock-after: <ISO time>" into .claude/EVAL_UNLOCK.')
    print(json.dumps({
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'deny',
            'permissionDecisionReason': reason,
        },
        'systemMessage': reason,
    }))
    return 0


if __name__ == '__main__':
    sys.exit(main())
