"""
live/strategy.py
Live signal generation wrapper.

Responsibilities:
- Load strategy parameters from config
- Call the appropriate signal function on the latest closed bar
- Return a signal dict (direction, atr, entry, stop, tp) or None
- Support multiple strategies selected by config["strategy"]["name"]
"""

# TODO: implement LiveStrategy class
