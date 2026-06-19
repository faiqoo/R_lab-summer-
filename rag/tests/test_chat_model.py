from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from hawki_langchain import ChatHAWKI, HAWKIModel


@pytest.fixture
def model() -> ChatHAWKI:
    return ChatHAWKI(api_token="test-token")


class TestMessageConversion:
    def test_human_message_maps_to_user_role(self, model: ChatHAWKI) -> None:
        result = model._convert_messages([HumanMessage(content="hi")])
        assert result == [{"role": "user", "content": {"text": "hi"}}]

    def test_system_message_maps_to_system_role(self, model: ChatHAWKI) -> None:
        result = model._convert_messages([SystemMessage(content="be concise")])
        assert result == [{"role": "system", "content": {"text": "be concise"}}]

    def test_ai_message_maps_to_assistant_role(self, model: ChatHAWKI) -> None:
        result = model._convert_messages([AIMessage(content="hello")])
        assert result == [{"role": "assistant", "content": {"text": "hello"}}]

    def test_multi_turn_order_preserved(self, model: ChatHAWKI) -> None:
        messages = [
            SystemMessage(content="be concise"),
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
            HumanMessage(content="how are you?"),
        ]
        result = model._convert_messages(messages)
        assert [r["role"] for r in result] == ["system", "user", "assistant", "user"]


def _make_response(text: str) -> MagicMock:
    mock = MagicMock(spec=httpx.Response)
    mock.json.return_value = {"success": True, "content": {"text": text}}
    mock.raise_for_status.return_value = None
    return mock


def _make_error_response(message: str = "Model not found") -> MagicMock:
    mock = MagicMock(spec=httpx.Response)
    mock.json.return_value = {"success": False, "message": message}
    mock.raise_for_status.return_value = None
    return mock


class TestGenerate:
    def test_returns_ai_message_on_success(self, model: ChatHAWKI) -> None:
        with patch("hawki_langchain.mixins.httpx.post") as mock_post:
            mock_post.return_value = _make_response("Hello!")
            result = model._generate([HumanMessage(content="say hello")])
        assert result.generations[0].message.content == "Hello!"

    def test_sends_correct_model_and_message(self, model: ChatHAWKI) -> None:
        with patch("hawki_langchain.mixins.httpx.post") as mock_post:
            mock_post.return_value = _make_response("ok")
            model._generate([HumanMessage(content="test input")])
        payload = mock_post.call_args.kwargs["json"]
        assert payload["payload"]["model"] == HAWKIModel.QWEN3_30B
        assert payload["payload"]["messages"][0]["content"]["text"] == "test input"

    def test_sends_bearer_token(self, model: ChatHAWKI) -> None:
        with patch("hawki_langchain.mixins.httpx.post") as mock_post:
            mock_post.return_value = _make_response("ok")
            model._generate([HumanMessage(content="test")])
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-token"

    def test_raises_value_error_on_api_failure(self, model: ChatHAWKI) -> None:
        with patch("hawki_langchain.mixins.httpx.post") as mock_post:
            mock_post.return_value = _make_error_response("Model not found")
            with pytest.raises(ValueError, match="HAWKI API error: Model not found"):
                model._generate([HumanMessage(content="test")])

    def test_raises_http_error_on_bad_status(self, model: ChatHAWKI) -> None:
        with patch("hawki_langchain.mixins.httpx.post") as mock_post:
            bad = MagicMock(spec=httpx.Response)
            bad.raise_for_status.side_effect = httpx.HTTPStatusError(
                "401 Unauthorized",
                request=MagicMock(),
                response=MagicMock(),
            )
            mock_post.return_value = bad
            with pytest.raises(httpx.HTTPStatusError):
                model._generate([HumanMessage(content="test")])


class TestStructuredOutput:
    def test_with_structured_output_parses_pydantic_model(
        self, model: ChatHAWKI
    ) -> None:
        from pydantic import BaseModel

        class Answer(BaseModel):
            value: str
            confidence: float

        json_response = '{"value": "Paris", "confidence": 0.99}'

        with patch("hawki_langchain.mixins.httpx.post") as mock_post:
            mock_post.return_value = _make_response(json_response)
            structured = model.with_structured_output(Answer)
            result = structured.invoke([HumanMessage(content="Capital of France?")])

        assert isinstance(result, Answer)
        assert result.value == "Paris"
        assert result.confidence == 0.99


class TestAGenerate:
    @pytest.mark.asyncio
    async def test_returns_ai_message_on_success(self, model: ChatHAWKI) -> None:
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=_make_response("Async reply"))

        async def fake_aenter(_: Any) -> MagicMock:
            return mock_client

        async def fake_aexit(*_: Any) -> bool:
            return False

        with patch("hawki_langchain.mixins.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = fake_aenter
            mock_cls.return_value.__aexit__ = fake_aexit
            result = await model._agenerate([HumanMessage(content="async test")])

        assert result.generations[0].message.content == "Async reply"
