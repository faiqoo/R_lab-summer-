"""All LLM prompts and canned responses used by the RAG query path.

HAWKI-hosted models ignore system messages, so every instruction lives in the
human turn. History placeholders come first so the model sees prior turns
before the current question.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            (
                "You are an assistant for the Berufearchiv archive at "
                "Uni Koblenz. Answer using only the provided archive "
                "documents. Cite the source in brackets after relevant "
                "statements. If the answer is not in the documents, say so "
                "clearly. You MUST answer in {language} — the same language "
                "as the question.\n\n"
                "Documents:\n{context}\n\n"
                "Question: {question}"
            ),
        ),
    ]
)

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            (
                "Given the conversation above, rewrite the following "
                "question into a self-contained search query that includes "
                "all necessary context. Output only the rewritten query, "
                "nothing else.\n\n"
                "Question: {question}"
            ),
        ),
    ]
)

FALLBACK_LOW_RELEVANCE = (
    "I could not find relevant documents in the Berufearchiv archive "
    "for this question. Please try rephrasing or ask about a specific occupation."
)
