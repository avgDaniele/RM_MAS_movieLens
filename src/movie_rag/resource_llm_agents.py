from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from pydantic import BaseModel

from .resource_domain import ResourceDomainProfile
from .resource_llm_models import (
    GroundedAnswer,
    LLMCallRecord,
    ResourceRetrievalPlan,
)
from .resource_models import ResourceSearchResponse


class TextGenerationProvider(Protocol):
    async def generate_text(
        self,
        *,
        agent: str,
        prompt_version: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int,
    ) -> Any: ...


def _trace_dict(trace: Any) -> dict[str, Any]:
    if trace is None:
        return {}
    if hasattr(trace, "model_dump"):
        return trace.model_dump(mode="json")
    if isinstance(trace, dict):
        return trace
    return {"repr": repr(trace)}


def _extract_json_text(text: str) -> str:
    text = str(text).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        return text[first:last + 1]
    return text


def _parse_model(text: str, schema: type[BaseModel]) -> BaseModel:
    payload = json.loads(_extract_json_text(text))
    return schema.model_validate(payload)


def _schema_instruction(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)


@dataclass
class ResourcePlannerAgent:
    provider: TextGenerationProvider
    domain: ResourceDomainProfile
    temperature: float | None = None
    max_output_tokens: int = 800
    prompt_version: str = "resource_planner_v1"
    query_rewrite_strategy: str = "none"

    async def run(self, request: str) -> tuple[ResourceRetrievalPlan, LLMCallRecord]:
        system_prompt = self._system_prompt()
        user_prompt = (
            f"USER REQUEST:\n{request.strip()}\n\n"
            "Return only one JSON object matching the schema below.\n"
            f"{_schema_instruction(ResourceRetrievalPlan)}"
        )
        kwargs = {
            "agent": "resource_planner",
            "prompt_version": self.prompt_version,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        result = await self.provider.generate_text(**kwargs)
        plan = _parse_model(result.text, ResourceRetrievalPlan)
        self.domain.validate_native_filters(plan.filters)
        return plan, LLMCallRecord(
            role="planner",
            prompt_version=self.prompt_version,
            provider_trace=_trace_dict(getattr(result, "trace", None)),
        )

    def _system_prompt(self) -> str:
        allowed = ", ".join(self.domain.allowed_filter_keys) or "none"
        extra = "\n".join(f"- {line}" for line in self.domain.planner_instructions)
        metadata = "\n".join(
            f"- {key}: {value}" for key, value in self.domain.metadata_hints.items()
        )
        if self.query_rewrite_strategy not in {"none", "translate_to_catalog_language"}:
            raise ValueError(
                "query_rewrite_strategy must be 'none' or 'translate_to_catalog_language'"
            )
        language_rule = (
            f"Write semantic_query in the catalog language ({self.domain.catalog_language}). "
            "Translate/rephrase the user's semantic intent faithfully; do not add constraints."
            if self.query_rewrite_strategy == "translate_to_catalog_language"
            else "Preserve the user's language in semantic_query unless a faithful reformulation requires otherwise."
        )
        return f"""You are the retrieval planner for {self.domain.display_name}.
The user is a {self.domain.user_role}. Your task is: {self.domain.task_description}
The searchable catalog is: {self.domain.catalog_description}

Produce a semantic query for vector retrieval and only metadata filters that are explicitly
supported and justified by the user's request. Never invent constraints.

Query-language rule: {language_rule}

Allowed native filter keys: {allowed}
Metadata hints:
{metadata or '- none'}

Rules:
- semantic_query should preserve semantic content, intent, topic, mood and descriptive concepts;
- numeric/categorical constraints should go in filters when an allowed key exists;
- do not choose an embedding model, vector database, candidate_k, reranker or retrieval profile;
- ask for clarification only when a missing detail prevents a meaningful search; otherwise search with what is known;
- rationale must be a short decision summary, not private chain-of-thought;
- output valid JSON only.
{extra}
"""


@dataclass
class ResourceAnswerAgent:
    provider: TextGenerationProvider
    domain: ResourceDomainProfile
    temperature: float | None = None
    max_output_tokens: int = 1200
    prompt_version: str = "resource_answer_v1"

    async def run(
        self,
        original_request: str,
        search: ResourceSearchResponse,
    ) -> tuple[GroundedAnswer, LLMCallRecord]:
        resources = [
            {
                "resource_id": hit.resource_id,
                "title": hit.title,
                "rank": hit.rank,
                "score": hit.score,
                "text": hit.text,
                "metadata": hit.metadata,
            }
            for hit in search.results
        ]
        system_prompt = self._system_prompt()
        user_prompt = (
            f"ORIGINAL REQUEST:\n{original_request.strip()}\n\n"
            "RETRIEVED RESOURCES (the only resources you may recommend):\n"
            f"{json.dumps(resources, ensure_ascii=False, indent=2)}\n\n"
            "Return only one JSON object matching the schema below.\n"
            f"{_schema_instruction(GroundedAnswer)}"
        )
        kwargs = {
            "agent": "resource_answer",
            "prompt_version": self.prompt_version,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        result = await self.provider.generate_text(**kwargs)
        answer = _parse_model(result.text, GroundedAnswer)
        return answer, LLMCallRecord(
            role="answer",
            prompt_version=self.prompt_version,
            provider_trace=_trace_dict(getattr(result, "trace", None)),
        )

    def _system_prompt(self) -> str:
        extra = "\n".join(f"- {line}" for line in self.domain.answer_instructions)
        return f"""You are the grounded answer agent for {self.domain.display_name}.
The resources come from a controlled catalog. Base every recommendation and factual claim
about a resource only on the retrieved resource payload provided by the application.

Rules:
- recommend only resource_ids present in the retrieved list;
- never invent resources, metadata or unseen content;
- explain relevance to the user's request using available evidence;
- if evidence is insufficient, say so;
- keep recommendations concise and useful;
- status must be 'ok' when recommendations are produced;
- output valid JSON only.
{extra}
"""


__all__ = [
    "TextGenerationProvider",
    "ResourcePlannerAgent",
    "ResourceAnswerAgent",
]
