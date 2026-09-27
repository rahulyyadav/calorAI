from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from calorai_agent.app import create_agent
from calorai_agent.config import Settings
from calorai_agent.domain import InboundMessage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with the CalorAI meal-logging agent")
    parser.add_argument("message", nargs="*", help="Send one message and exit")
    parser.add_argument("--user", help="Override CALORAI_USER_ID")
    parser.add_argument("--timezone", help="Override CALORAI_TIMEZONE")
    parser.add_argument(
        "--planner",
        choices=("auto", "deterministic", "model"),
        help="Override CALORAI_PLANNER (deterministic needs no API key)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    user_id = args.user or settings.default_user_id
    timezone = args.timezone or settings.default_timezone
    settings = replace(
        settings,
        default_user_id=user_id,
        default_timezone=timezone,
        planner=args.planner or settings.planner,
    )

    agent = create_agent(settings)

    if args.message:
        print(agent.invoke(user_id, " ".join(args.message), timezone=timezone))
        return

    print("CalorAI — describe a meal, ask for totals, or type 'quit'.")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            return
        response = agent.handle(
            InboundMessage(
                user_id=user_id,
                text=message,
                external_id=f"cli:{uuid4()}",
                channel="cli",
                timezone=timezone,
                received_at=datetime.now(UTC),
            )
        )
        print(f"calorai> {response}")


if __name__ == "__main__":
    main()
