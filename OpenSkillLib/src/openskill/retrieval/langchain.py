"""
LangChain Integration — OpenSkillRetriever
=========================================
Transforms OpenSkill into a LangChain BaseRetriever for use 
in RetrievalQA, ConversationalRetrievalChain, AgentExecutor, etc.

Usage:

    from langchain.chains import RetrievalQA
    from openskill.retrieval.langchain import OpenSkillRetriever

    retriever = OpenSkillRetriever(skill_dir="./skills")
    qa = RetrievalQA.from_chain_type(
        llm=Ollama(model="qwen2.5-coder"),
        retriever=retriever,
    )
    qa.invoke("How to implement Raft failover?")
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, AsyncIterator

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from openskill import OpenSkillClient, LocalDiskStore
from openskill.llm.ollama import OllamaProvider

if TYPE_CHECKING:
    pass


class OpenSkillRetriever(BaseRetriever):
    """
    LangChain BaseRetriever that searches for skills via TurboQuant + S-Path-RAG.

    The returned documents are normative constraints extracted from 
    the most relevant skills, formatted as LangChain Documents.
    """

    def __init__(
        self,
        skill_dir: str = "./skills_output",
        top_k: int = 3,
        format: str = "constraints",  # "constraints" | "full" | "summary"
        llm_api_key: str | None = None,
        use_ollama: bool = False,
        ollama_model: str = "qwen2.5-coder:7b",
        **client_kwargs,
    ):
        # Determines storage and LLM provider
        store = LocalDiskStore(skill_dir)

        if use_ollama:
            llm: Any = OllamaProvider(model=ollama_model)
        elif llm_api_key:
            from openskill.llm.openrouter import OpenRouterProvider
            llm = OpenRouterProvider(api_key=llm_api_key)
        else:
            llm = None  # Uses client default (OpenRouter or fails)

        self._client = OpenSkillClient(store=store, llm=llm, **client_kwargs)
        self.top_k = top_k
        self.format = format

    def _extract_constraints(self, markdown: str) -> str:
        """Extracts only Normative Constraints from markdown."""
        lines = []
        in_constraints = False
        for line in markdown.split("\n"):
            if "## Normative Constraints" in line or \
               "## Constraints" in line or \
               "## Reasoning Invariants" in line:
                in_constraints = True
                continue
            if in_constraints and line.startswith("## "):
                break
            if in_constraints and line.strip():
                # Removes markdown formatting from bullets
                cleaned = re.sub(r"^[-*]\s*", "", line)
                lines.append(cleaned)
        return "\n".join(lines) if lines else markdown[:500]

    def _skill_to_document(self, skill_data: dict) -> Document:
        """Converts a skill to a LangChain Document."""
        content = skill_data.get("content", "")
        meta = {
            "skill_id": skill_data.get("id", ""),
            "title": skill_data.get("title", ""),
            "category": skill_data.get("category", ""),
            "domain": skill_data.get("domain", ""),
        }

        if self.format == "constraints":
            page_content = self._extract_constraints(content)
        elif self.format == "summary":
            page_content = content[:300] + "..."
        else:
            page_content = content

        return Document(page_content=page_content, metadata=meta)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Synchronous version (LangChain sync runner)."""
        import asyncio
        return asyncio.run(self._aget_relevant_documents(query, run_manager))

    async def _aget_relevant_documents(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Async search via TurboQuant + S-Path-RAG."""
        result = await self._client.retrieve(query=query, top_k=self.top_k)

        docs = []
        for skill in result.get("skills", []):
            # Enrich skill with store content
            skill_md = await self._client.store.get_skill_md(skill.get("id", ""))
            if skill_md:
                skill["content"] = skill_md
            doc = self._skill_to_document(skill)
            docs.append(doc)

        # LangChain callback
        if run_manager:
            await run_manager.on_retriever_end(
                documents=docs,
                query=query,
            )

        return docs
