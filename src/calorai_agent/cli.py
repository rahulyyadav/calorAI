from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from calorai_agent.app import create_agent, prepare_runtime
from calorai_agent.config import Settings
from calorai_agent.domain import InboundMessage, MediaRef


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with the CalorAI meal-logging agent")
    parser.add_argument("message", nargs="*", help="Send one message and exit")
    parser.add_argument("--user", help="Override CALORAI_USER_ID")
    parser.add_argument("--timezone", help="Override CALORAI_TIMEZONE")
    parser.add_argument(
        "--image",
        metavar="PATH",
        help="Attach a jpeg, png, or webp photo of the plate (needs a vision model)",
    )
    parser.add_argument(
        "--planner",
        choices=("auto", "deterministic", "model"),
        help="Override CALORAI_PLANNER (deterministic needs no API key)",
    )
    parser.add_argument(
        "--logs",
        choices=("DEBUG", "INFO", "WARNING"),
        help="Show the structured trace on stderr instead of only the reply (default: WARNING)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    user_id = args.user or settings.default_user_id
    timezone = args.timezone or settings.default_timezone
    settings = prepare_runtime(
        replace(
            settings,
            default_user_id=user_id,
            default_timezone=timezone,
            planner=args.planner or settings.planner,
            log_level=args.logs or settings.log_level,
        )
    )

    agent = create_agent(settings)

    if args.message or args.image:
        media = None if args.image is None else photo_reference(args.image)
        print(agent.invoke(user_id, " ".join(args.message), timezone=timezone, media=media))
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


def photo_reference(path: str) -> MediaRef:
    """Identify a CLI photo by where it is, so the same file sent twice stays one message."""
    resolved = str(Path(path).expanduser().resolve())
    digest = hashlib.sha256(resolved.encode()).hexdigest()[:16]
    return MediaRef(external_id=f"cli-photo:{digest}", locator=resolved)


if __name__ == "__main__":
    main()
