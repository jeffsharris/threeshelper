# Human Watch Inbox

`observe_human_game.py` writes human/tracker observation sessions here. A
session should look like:

```text
datasets/human_watch/session_YYYYMMDD_HHMMSS/events.jsonl
```

After recording a strong game, process any new sessions with:

```bash
.venv/bin/python -m threes_rl.human_diagnostics_batch --run
```

The current research target is at least five independent games that reach a
non-starter `1536`, including one or more games that reach `3072`. Raw session
artifacts in this directory are intentionally git-ignored; this README is the
tracked landing note.
