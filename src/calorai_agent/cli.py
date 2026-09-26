from __future__ import annotations

import argparse

from calorai_agent.app import create_agent
from calorai_agent.config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with the CalorAI meal-logging agent")
    parser.add_argument("message", nargs="*", help="Send one message and exit")
    parser.add_argument("--user", help="Override CALORAI_USER_ID")
    parser.add_argument("--timezone", help="Override CALORAI_TIMEZONE")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    user_id = args.user or settings.default_user_id
    timezone = args.timezone or settings.default_timezone
    if user_id != settings.default_user_id or timezone != settings.default_timezone:
        settings = Settings(settings.database_path, user_id, timezone)
    agent = create_agent(settings)

    if args.message:
        print(agent.invoke(user_id, " ".join(args.message), timezone=timezone))
        return

    print("CalorAI Phase 1 — type 'quit' to exit")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if message.lower() in {"quit", "exit"}:
            return
        if message:
            print(f"calorai> {agent.invoke(user_id, message, timezone=timezone)}")


if __name__ == "__main__":
    main()
