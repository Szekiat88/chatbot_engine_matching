"""
Lightweight helpers for running Gemini via the Google ADK (Agents Development Kit).

The ADK runner streams tokens and reuses in-memory session/artifact stores, which
reduces per-request overhead compared to constructing a new genai client each time.
Functions here stay minimal so existing code can opt into ADK by choosing the
``provider="adk"`` path.
"""
from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Optional

from google.genai import types as genai_types


def _ensure_api_key() -> None:
    """Make sure ADK-compatible API key is exposed as GOOGLE_API_KEY."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = key
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY to use google-adk.")


@lru_cache()
def _get_runner():
    """Return a shared ADK runner with in-memory session/artifact stores."""
    _ensure_api_key()
    from google.adk import Runner
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.sessions import InMemorySessionService

    return Runner(
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
    )


def _build_agent(model: str, instruction: str):
    from google.adk import Agent

    return Agent(
        model=model,
        name="engine_matching_agent",
        instruction=instruction,
    )


async def _adk_run(prompt: str, *, model: str, instruction: str) -> str:
    """Run a single prompt through ADK and return the latest text chunk."""
    runner = _get_runner()
    agent = _build_agent(model, instruction)
    input_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(prompt)],
    )

    last_text: str = ""
    async for event in runner.run_stream(agent=agent, input=input_content):
        content = getattr(event, "content", None)
        if not content:
            continue
        for part in content.parts:
            text = getattr(part, "text", None)
            if text:
                last_text = text
    return last_text.strip()


def adk_generate_text(
    prompt: str,
    *,
    model: str,
    instruction: Optional[str] = None,
) -> str:
    """Synchronous wrapper for ADK text generation."""
    instruction = instruction or "You are a helpful assistant. Follow the prompt exactly."
    try:
        return asyncio.run(_adk_run(prompt, model=model, instruction=instruction))
    except RuntimeError as exc:
        if "event loop is running" in str(exc):
            raise RuntimeError(
                "adk_generate_text cannot be called from within an active asyncio loop; "
                "await _adk_run instead."
            ) from exc
        raise
