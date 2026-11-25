# From AURA-V5 src/rl_agent/utils/sync_hunter.py (commit 41007db), adapted for Spellcast3D:
#   - torch is imported inside hunt_syncs (module level is stdlib only, like the rest of spellcast/profiling/);
#   - a module logger (log = logging.getLogger(__name__)) for the one root logging.info call ("armed"); the hunter's
#     own per-pid logger and its report file are unchanged.
#   Otherwise verbatim. It reports TORCH-side synchronisations only: torch.cuda.set_sync_debug_mode("warn") warns
#   when a PyTorch operation makes the CPU wait for the GPU. For the torch-based tools (Pixal3D, LATO.2, TriFlow).
#   Qwen-Image runs no PyTorch operation at all (stable-diffusion.cpp drives the GPU through its own CUDA runtime), so
#   this would always report nothing there: its diagnostic_enable_sync_hunter switch uses cuda_trace_report.py's
#   sync_report instead (the CUDA runtime's own blocking calls and the GPU's idle gaps, from a CUPTI trace).
import logging
import os
import warnings
import linecache
import traceback
from contextlib import contextmanager

log = logging.getLogger(__name__)

# 1. Setup Persistent Logger (Module Level)
# Unique name per PID to ensure safety in Ray workers
LOGGER_NAME = f"sync_hunter_{os.getpid()}"
hunter_logger = logging.getLogger(LOGGER_NAME)
hunter_logger.propagate = False 
hunter_logger.setLevel(logging.WARNING)

# Configuration
LOG_FULL_STACK_TRACE = True

# One handler per log path, built ONCE per process: mode='w' truncates, so re-creating it on a
# second hunt_syncs() over the same path would erase that file's earlier content.
_FILE_HANDLERS = {}


def _handler_for_file(log_file):
    """Returns this path's FileHandler, creating it on first use. NOT attached here: hunt_syncs
    attaches on entry and detaches on exit, so each report file only ever receives its own
    phase. Attaching for the process lifetime meant every later phase's syncs were also written
    into every earlier phase's file."""
    key = os.path.abspath(log_file)
    if key in _FILE_HANDLERS:
        return _FILE_HANDLERS[key]
    try:
        dir_name = os.path.dirname(key)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fh = logging.FileHandler(key, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(message)s'))
        _FILE_HANDLERS[key] = fh
        return fh
    except Exception as e:
        print(f"SyncHunter failed to init log file {log_file}: {e}")
        raise

@contextmanager
def hunt_syncs(enabled=True, log_file=None):
    """
    Context manager to detect and log CUDA-CPU synchronizations.
    """
    if not enabled:
        yield
        return

    import torch

    if log_file is None:
        log_file = f"runs/sync_report_pid_{os.getpid()}.txt"

    # 1. Attach this file's handler for the duration of THIS context only.
    handler = _handler_for_file(log_file)
    hunter_logger.addHandler(handler)
    hunter_logger.warning(f">>> SYNC HUNTER INITIALIZED. PID: {os.getpid()}")
    log.info(f">>> SYNC HUNTER: armed -> {log_file}")

    # 2. Capture Original Warning Handler
    original_showwarning = warnings.showwarning
    cwd = os.getcwd()

    def custom_sync_handler(message, category, filename, lineno, file=None, line=None):
        """
        Intercepts warnings to check for CUDA synchronization triggers.
        """
        msg_str = str(message)
        
        # Filter unrelated warnings
        if "prototype feature" in msg_str: 
            return

        # Check for PyTorch Sync Warning (UserWarning class + specific text)
        is_sync_warning = issubclass(category, UserWarning) and ("synchroniz" in msg_str.lower())
        
        if not is_sync_warning:
            # Pass through non-sync warnings to the original handler
            if original_showwarning:
                original_showwarning(message, category, filename, lineno, file, line)
            return
        
        # --- LOGGING LOGIC ---
        try:
            clean_path = os.path.relpath(filename, cwd)
        except ValueError:
            clean_path = filename

        try:
            code_line = linecache.getline(filename, lineno).strip()
        except Exception:
            code_line = "<could not read source>"
            
        if LOG_FULL_STACK_TRACE:
            # [SAFETY FIX] Use extract_stack with limit to avoid deep object inspection overhead
            # and prevent potential interference with Autograd graph retention.
            try:
                # Limit depth to 15 frames to reduce overhead
                stack_summary = traceback.extract_stack(limit=15)
                # Filter out the sync_hunter frames themselves
                stack_summary = stack_summary[:-2] 
                formatted_stack = "".join(traceback.format_list(stack_summary))
            except Exception:
                formatted_stack = "<stack trace failed>"

            log_msg = (
                f"SYNC DETECTED\n"
                f"Trigger: {clean_path}:{lineno}\n"
                f"Code: {code_line}\n"
                f"Message: {msg_str}\n"
                f"Stack Trace:\n{formatted_stack}\n"
                f"--\n"
            )
        else:
            log_msg = f"{clean_path}:{lineno} :: {code_line}\n"

        hunter_logger.warning(log_msg)

        # Force flush
        for h in hunter_logger.handlers: 
            if hasattr(h, 'flush'): h.flush()

    # 3. Enable Debug Mode
    try:
        # Report every occurrence, not the first per call site per process: the default action
        # dedups through each module's __warningregistry__, which outlives this context. Scoped
        # to the sync message, so the process's other warning filters stay as they are.
        with warnings.catch_warnings():
            warnings.filterwarnings("always", message=".*synchroniz", category=UserWarning)
            warnings.showwarning = custom_sync_handler

            # Only set mode if CUDA is actually available
            if torch.cuda.is_available():
                torch.cuda.set_sync_debug_mode("warn")

            yield

    finally:
        # 4. Restore State (Critical); catch_warnings restores filters and showwarning.
        if torch.cuda.is_available():
            torch.cuda.set_sync_debug_mode("default")

        warnings.showwarning = original_showwarning
        handler.flush()
        hunter_logger.removeHandler(handler)
