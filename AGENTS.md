# Internal engineering toolkit

- CPU-first. Never download models/data or start GPU inference automatically.
- `run` and `compose` require explicit opt-in, concrete local paths, runtime, output and budget.
- Preserve frozen raw evidence; derive into new outputs. Atomic mutable progress is separate.
- Signal only owned subprocess groups. Never pkill by name, kill self, or stop unknown GPU work.
- Keep this repository PRIVATE; verify owner/visibility/destination before every push or asset upload.
- Do not change upstream research workspaces/environments. No PyPI, public release, telemetry or CI auto-run.
- Keep original licenses, hashes, historical failures, UNKNOWN and INCONCLUSIVE states.
