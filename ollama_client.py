#!/usr/bin/env python3
"""Ollama client for interacting with local Ollama models.

This script provides a simple interface to send prompts to the local Ollama API
and receive responses from the configured model.

Where the daemon is and how long to wait for it are read from
`langgraph_agent.config` rather than respelled here, so a daemon moved with
`OLLAMA_BASE_URL` -- the variable `.env.example` documents -- moves this script
with it. The raw client honours `OLLAMA_HOST` instead, which this project sets
nowhere, so a script left to that default keeps talking to 127.0.0.1 while every
seat follows the move.
"""

import sys

import ollama

from langgraph_agent.config import (
    LLM_TIMEOUT_SECONDS,
    _ollama_base_url,
    free_the_cards_for,
    is_local_ollama_model,
)
from langgraph_agent.control import GPU_ARBITER

# The same local tag the Architect/Planner/Researcher seats default to
# (config.DEFAULT_SEATS): 5.3 GB, which loads 100% onto the cards on its own
# (`qwen3.8:latest`, at 17 GB against 6 GB of VRAM, never does). It does not
# fit beside `qwen3-embedding:latest` -- the two want roughly 10 GB of that
# 6 GB, so this script takes `GPU_ARBITER` and evicts whatever else is
# resident first, exactly as the seats do.
DEFAULT_MODEL = "hf.co/mradermacher/dolphin-2.9.1-yi-1.5-9b-GGUF:Q4_K_M"


def send_prompt(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a user prompt to the local Ollama API and return the response.

    Args:
        prompt: The user's input prompt.
        model: The model name to use. Defaults to `DEFAULT_MODEL`.

    Returns:
        The model's response text, and "" when the reply carried none: a
        thinking model puts its answer in `thinking` and leaves `content` unset,
        so this is typed `str | None` upstream and must not be returned raw.

    The client is built here with an explicit timeout instead of calling
    `ollama.chat()`, which uses the module client -- and that one defaults to no
    deadline at all, so a wedged daemon blocks forever with nothing raised for
    any caller to catch.
    """
    client = ollama.Client(host=_ollama_base_url(), timeout=LLM_TIMEOUT_SECONDS)
    # One model on the cards at a time, the same way every seat takes them.
    with GPU_ARBITER.exclusive(f"script:{model}"):
        if is_local_ollama_model(model):
            free_the_cards_for(model)
        reply = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    return reply.message.content or ""


if __name__ == "__main__":
    # Demonstrate a round-trip request/response cycle
    test_prompt = "Hello, how are you?"
    print(f"Sending prompt: {test_prompt}")
    print(f"Using model: {DEFAULT_MODEL}")
    print(f"Talking to: {_ollama_base_url()}")
    print("-" * 40)

    try:
        response = send_prompt(test_prompt)
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        print(f"Make sure the Ollama server is running ({_ollama_base_url()})")
        # Non-zero, or every reader of an exit code -- an install smoke check, a
        # CI step, a Builder's `terminal_execute`, whose status the Architect
        # then rules on -- records a dead daemon as a working round-trip.
        sys.exit(1)

    print("Model response:")
    print(response or "(the model returned no content)")
