"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Header, HTTPException, Request


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Reject the request when an API key is configured and not provided."""
    expected = request.app.state.api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
