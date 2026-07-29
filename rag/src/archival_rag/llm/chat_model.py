from __future__ import annotations

import builtins
import re
from enum import StrEnum
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import BaseModel, SecretStr

from archival_rag.llm.client import HAWKIClientMixin


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced top-level JSON object in ``text``, or ``None``.

    Tolerates minor prose or a ```json fence around the object, which some
    HAWKI-hosted models emit despite a JSON-only instruction.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class HAWKIModel(StrEnum):
    LLAMA_8B = "meta-llama-3.1-8b-instruct"
    LLAMA_70B = "llama-3.3-70b-instruct"
    QWEN3_30B = "qwen3-30b-a3b-instruct-2507"
    QWEN3_CODER = "qwen3-coder-30b-a3b-instruct"
    QWEN3_OMNI = "qwen3-omni-30b-a3b-instruct"
    GEMMA_31B = "gemma-4-31b-it"
    DEVSTRAL = "devstral-2-123b-instruct-2512"
    APERTUS_70B = "apertus-70b-instruct-2509"
    OSS_120B = "openai-gpt-oss-120b"


class ChatHAWKI(HAWKIClientMixin, BaseChatModel):
    api_url: str = "https://chat.edu-ki-rlp.de"
    api_token: SecretStr
    model: str = HAWKIModel.QWEN3_30B

    @property
    def _llm_type(self) -> str:
        return "hawki"

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        role_map: dict[type, str] = {
            SystemMessage: "system",
            HumanMessage: "user",
            AIMessage: "assistant",
        }
        return [
            {
                "role": role_map.get(type(msg), "user"),
                "content": {"text": str(msg.content)},
            }
            for msg in messages
        ]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "payload": {
                "model": self.model,
                "messages": self._convert_messages(messages),
            }
        }
        data = self._post_sync("/api/ai-req", payload)
        text: str = data["content"]["text"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def with_structured_output(
        self,
        schema: builtins.dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            output_parser: Runnable[Any, Any] = PydanticOutputParser(
                pydantic_object=schema
            )
        else:
            output_parser = JsonOutputParser()

        from langchain_core.runnables import RunnableLambda

        instructions = (
            output_parser.get_format_instructions()
            if hasattr(output_parser, "get_format_instructions")
            else "Respond with valid JSON only."
        )
        suffix = (
            "\n\nReturn ONLY a single valid JSON object and nothing else: "
            "no prose, no explanation, no markdown, no code fences.\n" + instructions
        )

        def _inject_instructions(
            messages: list[BaseMessage],
        ) -> list[BaseMessage]:
            # HAWKI does not reliably honor the system role, so append the
            # format instructions to the last human turn instead of prepending
            # a SystemMessage (which the hosted models ignore).
            msgs = list(messages)
            for i in range(len(msgs) - 1, -1, -1):
                if isinstance(msgs[i], HumanMessage):
                    msgs[i] = HumanMessage(content=str(msgs[i].content) + suffix)
                    return msgs
            msgs.append(HumanMessage(content=suffix.strip()))
            return msgs

        def _clean(message: BaseMessage) -> str:
            # Tolerate minor prose around the JSON before the strict parser.
            text = str(getattr(message, "content", message))
            return _extract_json_object(text) or text

        return (
            RunnableLambda(_inject_instructions)
            | self
            | RunnableLambda(_clean)
            | output_parser
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "payload": {
                "model": self.model,
                "messages": self._convert_messages(messages),
            }
        }
        data = await self._post_async("/api/ai-req", payload)
        text: str = data["content"]["text"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
