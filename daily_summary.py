import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

CONFIG_PATH = Path(__file__).parent / "config.json"
LOG_PATH    = Path(__file__).parent / "trading_log.txt"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def send_telegram(token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
    resp.raise_for_status()


def parse_log_today(today_str: str):
    signals, placed, errors, criticals = [], [], [], []

    if not LOG_PATH.exists():
        return signals, placed, errors, criticals

    with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith(today_str):
                continue
            if "SIGNAL:" in line:
                m = re.search(r"\[(\w+#?)\].*SIGNAL:\s+(\w+)", line)
                if m:
                    signals.append(f"{m.group(1)} {m.group(2).upper()}")
            elif "Placing order:" in line:
                m = re.search(r"\[(\w+#?)\].*Placing order: (\w+ [\d.]+ lots)", line)
                if m:
                    placed.append(f"{m.group(1)} {m.group(2)}")
            elif "[ERROR]" in line:
                errors.append(line.strip())
            elif "[CRITICAL]" in line:
                criticals.append(line.strip())

    return signals, placed, errors, criticals


def read_states(symbols):
    open_positions, pending_orders = [], []
    for sym in symbols:
        path = Path(__file__).parent / f"state_{sym}.json"
        if not path.exists():
            continue
        with open(path) as f:
            state = json.load(f)
        if state.get("open_trade"):
            t = state["open_trade"]
            open_positions.append(
                f"  {sym} {t.get('direction','').upper()} @ {t.get('entry', '?')}"
            )
        elif state.get("pending_limit"):
            p = state["pending_limit"]
            open_positions.append(
                f"  {sym} {p.get('direction','').upper()} @ {p.get('price', '?')} (pending)"
            )
    return open_positions


def build_message(config, today_str, signals, placed, errors, criticals, positions):
    date_fmt = datetime.now(timezone.utc).strftime("%d %b %Y")
    start    = config.get("session_start_utc", 3)
    end      = config.get("session_end_utc", 20)

    lines = [
        f"Trading Bot — Daily Summary",
        f"{date_fmt} | Session {start:02d}:00–{end:02d}:00 UTC",
        "",
    ]

    if signals:
        lines.append(f"Signals today: {len(signals)}")
        for s in signals:
            lines.append(f"  {s}")
    else:
        lines.append("Signals today: 0")

    lines.append(f"Orders placed: {len(placed)}")
    lines.append(f"Errors: {len(errors)}")
    lines.append("")

    if positions:
        lines.append("Open positions:")
        lines += positions
    else:
        lines.append("Open positions: None")

    if criticals:
        lines.append("")
        lines.append("CRITICAL EVENTS:")
        for c in criticals:
            lines.append(f"  {c}")

    return "\n".join(lines)


def main():
    config   = load_config()
    token    = config.get("telegram_bot_token", "")
    chat_id  = config.get("telegram_chat_id", "")

    if not token or not chat_id:
        print("ERROR: telegram_bot_token and telegram_chat_id must be set in config.json")
        sys.exit(1)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    signals, placed, errors, criticals = parse_log_today(today_str)
    positions = read_states(config.get("symbols", []))
    message   = build_message(config, today_str, signals, placed, errors, criticals, positions)

    send_telegram(token, chat_id, message)
    print("Summary sent.")
    print(message)


if __name__ == "__main__":
    main()
