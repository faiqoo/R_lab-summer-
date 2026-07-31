from __future__ import annotations

from typing import Any

import httpx
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)


def _is_retryable(exc: BaseException) -> bool:
    """Retry HAWKI 5xx responses and transient transport errors, never 4xx."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


# HAWKI is intermittently flaky (5xx, dropped connections); back off with jitter.
_hawki_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=1.0, max=20.0),
    stop=stop_after_attempt(4),
    reraise=True,
)


class HAWKIClientMixin:
    """Reusable HTTP transport for the HAWKI API.

    Expects concrete classes to provide `api_url: str` and `api_token: SecretStr`.
    """

    api_url: str
    api_token: SecretStr

    def _build_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token.get_secret_value()}"}

    @staticmethod
    def _check(response: httpx.Response) -> dict[str, Any]:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # raise_for_status()'s message omits the response body, which is
            # where HAWKI actually says *why* a 4xx was rejected (e.g. which
            # field was invalid) — without it, a 422 shows up in logs as a
            # bare status code with no way to diagnose it after the fact.
            raise httpx.HTTPStatusError(
                f"{exc}\nResponse body: {response.text}",
                request=exc.request,
                response=exc.response,
            ) from None
        data: dict[str, Any] = response.json()
        if not data.get("success"):
            raise ValueError(f"HAWKI API error: {data.get('message', 'unknown')}")
        return data

    @_hawki_retry
    def _post_sync(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            f"{self.api_url}{endpoint}",
            headers=self._build_headers(),
            json=payload,
            timeout=60.0,
        )
        return self._check(response)

    @_hawki_retry
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
        return self._check(response)
