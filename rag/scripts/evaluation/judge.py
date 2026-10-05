"""LLM-as-judge wrapper: ChatHAWKI behind deepeval's DeepEvalBaseLLM interface.

Every eval metric call goes through HAWKI — never OpenAI or any external
provider. This is a hard project constraint: archival data and queries stay
on EU/university infrastructure. Deliberately uses a different model than
the main chat pipeline (settings.rag_eval_model, not settings.hawki_model) —
grading with the same model that generated the answer biases scores upward.
"""

from __future__ import annotations

from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from archival_rag.config import Settings
from archival_rag.llm import ChatHAWKI


class HAWKIEvalModel(DeepEvalBaseLLM):  # type: ignore[misc]  # base is Any (follow_imports=skip)
    """Judge model for deepeval metrics, backed by ChatHAWKI.

    Reuses ChatHAWKI.with_structured_output(), which already handles HAWKI's
    markdown-fence-wrapped JSON quirk — no new parsing logic needed here.
    """

    def __init__(self, settings: Settings, model_name: str | None = None) -> None:
        if settings.hawki_token is None:
            raise ValueError("HAWKI_TOKEN is required to run the eval harness")
        self._model_name = model_name or settings.rag_eval_model
        self._chat = ChatHAWKI(
            api_url=settings.hawki_api_url,
            api_token=settings.hawki_token,
            model=self._model_name,
        )

    def load_model(self) -> ChatHAWKI:
        return self._chat

    def get_model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        messages = [HumanMessage(content=prompt)]
        if schema is not None:
            chain = self._chat.with_structured_output(schema)
            return chain.invoke(messages)
        return str(self._chat.invoke(messages).content)

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> Any:
        messages = [HumanMessage(content=prompt)]
        if schema is not None:
            chain = self._chat.with_structured_output(schema)
            return await chain.ainvoke(messages)
        response = await self._chat.ainvoke(messages)
        return str(response.content)
