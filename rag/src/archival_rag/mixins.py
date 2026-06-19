from __future__ import annotations

from typing import Any

import httpx
from pydantic import SecretStr


class HAWKIClientMixin:
    """Reusable HTTP transport for the HAWKI API.

    Expects concrete classes to provide `api_url: str` and `api_token: SecretStr`.
    """

    api_url: str
    api_token: SecretStr

    def _build_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token.get_secret_value()}"}

    def _post_sync(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            f"{self.api_url}{endpoint}",
            headers=self._build_headers(),
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        if not data.get("success"):
            raise ValueError(f"HAWKI API error: {data.get('message', 'unknown')}")
        return data

    async def _post_async(
        self, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}{endpoint}",
                headers=self._build_headers(),
                json=payload,
                timeout=60.0,
            )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        if not data.get("success"):
            raise ValueError(f"HAWKI API error: {data.get('message', 'unknown')}")
        return data
