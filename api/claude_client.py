"""Anthropic Claude API — shared text completion helper."""
import os
from dotenv import load_dotenv

load_dotenv()

_client = None


def get_anthropic_client():
    global _client
    if _client is None:
        import anthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment")
        _client = anthropic.Anthropic(api_key=key)
    return _client


def get_claude_model() -> str:
    # Default: Claude Sonnet 4.6 (see Anthropic models overview). Override with ANTHROPIC_MODEL.
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def claude_complete(
    user_prompt: str,
    *,
    max_tokens: int = 2048,
    system: str | None = None,
) -> str:
    """Run a single user message through Claude; return plain text (no JSON parsing)."""
    client = get_anthropic_client()
    kwargs = {
        "model": get_claude_model(),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if system:
        kwargs["system"] = system
    msg = client.messages.create(**kwargs)
    parts = []
    for block in msg.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts).strip()
