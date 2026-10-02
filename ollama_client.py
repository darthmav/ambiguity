#!/usr/bin/env python3
"""One prompt to the local Ollama daemon, the way every seat sends one.

Where the daemon is and how long to wait for it come from
`langgraph_agent.config`, so a daemon moved with `OLLAMA_BASE_URL` moves this
script too; the raw client would follow `OLLAMA_HOST` instead, which this
project sets nowhere. The call takes `GPU_ARBITER` and evicts whatever else is
resident, as the seats do, and goes through the daemon's circuit, retried
briefly while the daemon cannot be reached.
"""

import sys

import ollama

from langgraph_agent.config import (
    DEFAULT_SEATS,
    LLM_TIMEOUT_SECONDS,
    OLLAMA_DAEMON,
    free_the_cards_for,
    is_local_ollama_model,
    ollama_base_url,
    retry_unreachable,
)
from langgraph_agent.control import GPU_ARBITER

# The Architect's default: a local tag that loads wholly onto the cards.
DEFAULT_MODEL = DEFAULT_SEATS["architect"]["model"]


def send_prompt(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send `prompt` to `model` and return the reply's text.

    "" when the reply carried none: a thinking model can answer in `thinking`
    and leave `content` unset. The client is built with an explicit timeout,
    because the module-level `ollama.chat()` waits forever on a wedged daemon.
    """
    client = ollama.Client(host=ollama_base_url(), timeout=LLM_TIMEOUT_SECONDS)
    messages = [{"role": "user", "content": prompt}]
    with GPU_ARBITER.exclusive(f"script:{model}"):
        if is_local_ollama_model(model):
            free_the_cards_for(model)
        reply = retry_unreachable(
            lambda: OLLAMA_DAEMON.call(client.chat, model=model, messages=messages),
            name=f"script:{model}",
        )
    return reply.message.content or ""


if __name__ == "__main__":
    test_prompt = "Hello, how are you?"
    print(f"Sending prompt: {test_prompt}")
    print(f"Using model: {DEFAULT_MODEL}")
    print(f"Talking to: {ollama_base_url()}")
    print("-" * 40)

    try:
        response = send_prompt(test_prompt)
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        print(f"Make sure the Ollama server is running ({ollama_base_url()})")
        # Non-zero, so anything reading the exit status does not record a dead
        # daemon as a working round-trip.
        sys.exit(1)

    print("Model response:")
    print(response or "(the model returned no content)")
