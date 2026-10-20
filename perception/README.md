# Perception service

The optional `--record DIR` mode writes a JSONL session, and `--record-video`
adds raw camera video. Recordings contain faces and other identifying visual
information. They must not be shared without the participant's consent. Keep
recordings under the gitignored `data/` directory and do not commit or deploy them.

Replay a session with `python perception/replay.py --log data/sessions/session_....jsonl`.
Use `--check` to compare recorded confirmations with the shared gate implementation.
