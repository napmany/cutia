"""Logging utilities for CUTIA."""

import logging
import sys


class CUTIALoggingStream:
    """
    A Python stream for use with event logging APIs throughout CUTIA.
    This stream wraps `sys.stderr`, forwarding `write()` and `flush()` calls
    to the stream referred to by `sys.stderr` at the time of the call.
    It also provides capabilities for disabling the stream to silence event logs.
    """

    def __init__(self):
        self._enabled = True

    def write(self, text):
        if self._enabled:
            sys.stderr.write(text)

    def flush(self):
        if self._enabled:
            sys.stderr.flush()

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value


CUTIA_LOGGING_STREAM = CUTIALoggingStream()


def disable_logging():
    """
    Disables the `CUTIALoggingStream` used by event logging APIs throughout CUTIA,
    silencing all subsequent event logs.
    """
    CUTIA_LOGGING_STREAM.enabled = False


def enable_logging():
    """
    Enables the `CUTIALoggingStream` used by event logging APIs throughout CUTIA,
    emitting all subsequent event logs. This reverses the effects of `disable_logging()`.
    """
    CUTIA_LOGGING_STREAM.enabled = True


def configure_cutia_loggers(root_module_name):
    """
    Configure CUTIA loggers with INFO level and clean formatting.

    Args:
        root_module_name: The root module name (e.g., "cutia")
    """
    # Simple formatter - just the message
    formatter = logging.Formatter(fmt="%(message)s")

    cutia_handler_name = "cutia_handler"
    handler = logging.StreamHandler(stream=CUTIA_LOGGING_STREAM)
    handler.setFormatter(formatter)
    handler.set_name(cutia_handler_name)

    logger = logging.getLogger(root_module_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handler if present (avoid duplicates)
    for existing_handler in logger.handlers[:]:
        if getattr(existing_handler, "name", None) == cutia_handler_name:
            logger.removeHandler(existing_handler)

    logger.addHandler(handler)
