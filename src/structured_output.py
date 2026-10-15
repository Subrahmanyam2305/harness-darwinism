"""Normalize narrowly permitted wrappers around Pydantic JSON responses."""

from __future__ import annotations


def unwrap_json_code_fence(response: str) -> str:
    """Remove one complete JSON code fence while leaving all other text intact."""
    stripped = response.strip()
    lines = stripped.splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```json", "```"}
        and lines[-1].strip() == "```"
    ):
        return "\n".join(lines[1:-1]).strip()
    return stripped
