# Derived from inference-lab; import/path boundaries changed for internal release01.
"""Independent lifecycle evidence; an API exit does not certify worker exits."""
def observed_shutdown(exit_record, log, internal_timeout):
    forced = [line for line in log.splitlines() if "force killing" in line.lower()]
    semaphore = [line for line in log.splitlines() if "leaked semaphore" in line]
    selected = [line for line in log.splitlines()
                if "Using " in line and ("Kernel" in line or "backend" in line or "FlashAttention" in line)]
    return {
        "parent_returncode": exit_record["returncode"],
        "shutdown_evidence": {"configured_internal_timeout_seconds": internal_timeout,
                              "forced_engine_kill_logged": bool(forced), "forced_kill_lines": forced,
                              "semaphore_warning_logged": bool(semaphore), "semaphore_lines": semaphore,
                              "engine_teardown_logged": "starting resource teardown" in log},
        "owned_process_cleanup": exit_record.get("owned_process_cleanup", "UNKNOWN"),
        "remaining_owned_pids": exit_record.get("remaining_owned_pids"),
        "worker_native_exit": {"status": "UNKNOWN", "returncode": None,
                               "reason": "Parent lifecycle and logs do not expose each native worker exit code"},
        "kernel_observation": {"level": "kernel_selected_log" if selected else "NOT_VERIFIED",
                               "selection_lines": selected, "profiler_verified": False},
    }
