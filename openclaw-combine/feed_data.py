#!/usr/bin/env python3
"""Feed seed conversation data to the OpenClaw-Combine API server.

Each seed entry describes a two-turn interaction:
  Turn 1 (main): user question  → model generates answer R1
  Turn 2 (main, session_done):  messages = [question, R1, challenge]
                                  → challenge becomes next_state for Turn 1

When the challenge arrives as next_state, the server concurrently:
  - Runs PRM eval  → ±1 reward  (Binary RL signal)
  - Runs hint judge → hint text  (OPD distillation signal)

Three possible outcomes per session:
  hint accepted AND eval ±1  → Combined OPD+RL sample
  hint accepted only          → OPD-only sample
  eval ±1 only                → RL-only sample

Usage
-----
# Start training first, then run this script in a separate terminal:
python feed_data.py                          # defaults: 30000, 16 target samples
python feed_data.py --port 30000 --target 32 --concurrency 8
python feed_data.py --data-file data/combine_seed_data.jsonl --loop
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_DATA_FILE = Path(__file__).parent / "data" / "combine_seed_data.jsonl"
_CHAT_ENDPOINT = "/v1/chat/completions"


def _load_seed_data(path: str) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    logger.info("Loaded %d seed entries from %s", len(entries), path)
    return entries


async def _chat(
    client: httpx.AsyncClient,
    base_url: str,
    messages: list[dict],
    session_id: str,
    turn_type: str = "main",
    session_done: bool = False,
    model: str = "qwen3-4b",
    api_key: str = "",
) -> str | None:
    """Send one chat request to the proxy and return the assistant text."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "X-Session-Id": session_id,
        "X-Turn-Type": turn_type,
    }
    if session_done:
        headers["X-Session-Done"] = "true"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {"model": model, "messages": messages}
    try:
        resp = await client.post(
            base_url + _CHAT_ENDPOINT, json=body, headers=headers, timeout=180.0
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        return msg.get("content") or ""
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "session=%s HTTP %d: %s", session_id, exc.response.status_code, exc.response.text[:300]
        )
    except Exception as exc:
        logger.warning("session=%s request error: %s", session_id, exc)
    return None


async def _run_session(
    client: httpx.AsyncClient,
    base_url: str,
    entry: dict,
    session_id: str,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    stats: dict,
) -> bool:
    """Execute one two-turn session. Returns True on success."""
    question: str = entry["question"]
    challenge: str = entry["challenge"]
    expected: str = entry.get("expected_signal", "unknown")

    async with semaphore:
        # ── Turn 1: send user question, get model answer ──────────────────
        turn1_messages = [{"role": "user", "content": question}]
        logger.info(
            "[%s] Turn 1 | expected=%s | Q: %s…", session_id[:8], expected, question[:60]
        )
        answer = await _chat(
            client, base_url, turn1_messages, session_id,
            turn_type="main", session_done=False, model=model, api_key=api_key,
        )
        if answer is None:
            logger.warning("[%s] Turn 1 failed, skipping session", session_id[:8])
            stats["failed"] += 1
            return False

        logger.info("[%s] Turn 1 answer: %s…", session_id[:8], answer[:80])

        # ── Turn 2: send [question, answer, challenge], mark session done ─
        turn2_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": challenge},
        ]
        logger.info("[%s] Turn 2 | challenge: %s…", session_id[:8], challenge[:60])
        answer2 = await _chat(
            client, base_url, turn2_messages, session_id,
            turn_type="main", session_done=True, model=model, api_key=api_key,
        )
        if answer2 is None:
            logger.warning("[%s] Turn 2 failed", session_id[:8])
            stats["failed"] += 1
            return False

        logger.info("[%s] Turn 2 answer: %s…", session_id[:8], answer2[:80])
        stats["success"] += 1
        return True


async def _run_until_target(
    base_url: str,
    seed_data: list[dict],
    target: int,
    concurrency: int,
    model: str,
    api_key: str,
    loop_seed: bool,
) -> None:
    """Keep feeding sessions until *target* sessions succeed (or data exhausted)."""
    semaphore = asyncio.Semaphore(concurrency)
    stats = {"success": 0, "failed": 0}
    tasks: list[asyncio.Task] = []
    data_idx = 0
    start = time.time()

    async with httpx.AsyncClient() as client:
        while stats["success"] < target:
            if data_idx >= len(seed_data):
                if loop_seed:
                    logger.info("Seed data exhausted, looping from beginning.")
                    data_idx = 0
                else:
                    logger.info("Seed data exhausted. Waiting for running sessions…")
                    break

            entry = seed_data[data_idx]
            data_idx += 1
            # Use entry's session_id as prefix + uuid suffix for uniqueness.
            session_id = f"{entry.get('session_id', 'sess')}-{uuid.uuid4().hex[:6]}"
            task = asyncio.create_task(
                _run_session(
                    client, base_url, entry, session_id,
                    model, api_key, semaphore, stats,
                )
            )
            tasks.append(task)

            # Don't saturate: if concurrency slots are all taken, wait a bit.
            await asyncio.sleep(0.05)

        # Wait for remaining in-flight sessions.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start
    logger.info(
        "Done: %d successful sessions, %d failed, %.1fs elapsed",
        stats["success"], stats["failed"], elapsed,
    )
    if stats["success"] < target:
        logger.warning(
            "Only %d/%d target sessions succeeded. "
            "Increase --loop or add more seed data.",
            stats["success"], target,
        )


def _wait_for_server(base_url: str, timeout: int = 120) -> bool:
    """Poll /healthz until the proxy is ready."""
    deadline = time.time() + timeout
    logger.info("Waiting for proxy server at %s …", base_url)
    while time.time() < deadline:
        try:
            r = httpx.get(base_url + "/healthz", timeout=5)
            if r.status_code == 200:
                logger.info("Proxy server is ready.")
                return True
        except Exception:
            pass
        time.sleep(3)
    logger.error("Proxy server did not become ready within %ds", timeout)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feed two-turn seed data to OpenClaw-Combine API server."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=30000, help="Proxy port (default: 30000)")
    parser.add_argument(
        "--data-file",
        default=str(_DEFAULT_DATA_FILE),
        help="Path to JSONL seed data file",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=16,
        help="Target number of successfully submitted sessions (default: 16, matches rollout_batch_size)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent sessions (default: 4)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("SERVED_MODEL_NAME", "qwen3-4b"),
        help="Model name to request (default: env SERVED_MODEL_NAME or 'qwen3-4b')",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("SGLANG_API_KEY", ""),
        help="API key for the proxy (default: env SGLANG_API_KEY)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop seed data from beginning when exhausted (useful for large targets)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Skip waiting for proxy server readiness check",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    if not args.no_wait:
        if not _wait_for_server(base_url):
            sys.exit(1)

    if not Path(args.data_file).exists():
        logger.error("Data file not found: %s", args.data_file)
        sys.exit(1)

    seed_data = _load_seed_data(args.data_file)
    if not seed_data:
        logger.error("No seed data loaded.")
        sys.exit(1)

    logger.info(
        "Starting feed: target=%d sessions, concurrency=%d, model=%s",
        args.target, args.concurrency, args.model,
    )
    asyncio.run(
        _run_until_target(
            base_url=base_url,
            seed_data=seed_data,
            target=args.target,
            concurrency=args.concurrency,
            model=args.model,
            api_key=args.api_key,
            loop_seed=args.loop,
        )
    )


if __name__ == "__main__":
    main()
