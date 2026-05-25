"""
live/logger.py
Structured logging setup for the live trading engine.

Responsibilities:
- Configure root logger with UTC timestamps
- Dual output: rotating file handler (logs/) and console handler
- Provide a get_logger(name) helper used by all live modules
- Ensure log directory exists on first run
"""

# TODO: implement setup_logger() and get_logger()
