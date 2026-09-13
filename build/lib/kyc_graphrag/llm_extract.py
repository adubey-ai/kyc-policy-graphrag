"""Optional model-based relation extraction with strict schema validation."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from urllib.request import Request, urlopen

from kyc_graphrag.corpus import Document, sentences
from kyc_graphrag.extract import Triple
from kyc_graphrag.schema import ENTITIES, RELATION_CUES

ENTITY_TYPES = {entity.canonical: entity.entity_type for entity in ENTITIES}
RELATIONS = {row[3] for row in RELATION_CUES}


@dataclass
class OpenAIRelationExtractor:
    """Extract typed triples using JSON Schema, then reject out-of-schema output."""

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    transport: Callable[[Request], bytes] | None = None

    def extract(self, docs: list[Document]) -> list[Triple]:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key and self.transport is None:
            raise RuntimeError("OPENAI_API_KEY is required for model extraction")
        triples: list[Triple] = []
        for sentence in sentences(docs):
            payload = self._request(sentence.text, key or "")
            for row in payload.get("triples", []):
                triple = self._validate(row, sentence.doc_id, sentence.sent_id, sentence.text)
                if triple is not None:
                    triples.append(triple)
        return list(dict.fromkeys(triples))

    def _request(self, text: str, key: str) -> dict:
        entity_names = sorted(ENTITY_TYPES)
        schema = {
            "name": "policy_triples",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "triples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "head": {"type": "string", "enum": entity_names},
                                "relation": {"type": "string", "enum": sorted(RELATIONS)},
                                "tail": {"type": "string", "enum": entity_names},
                            },
                            "required": ["head", "relation", "tail"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["triples"],
                "additionalProperties": False,
            },
        }
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Extract only explicit banking-policy relations. "
                            "Do not infer permissions that are not stated."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                "response_format": {"type": "json_schema", "json_schema": schema},
                "temperature": 0,
            }
        ).encode()
        request = Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        if self.transport is not None:
            raw = self.transport(request)
        else:
            with urlopen(request, timeout=30) as response:
                raw = response.read()
        result = json.loads(raw)
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)

    @staticmethod
    def _validate(
        row: dict,
        doc_id: str,
        sent_id: str,
        evidence: str,
    ) -> Triple | None:
        head = row.get("head")
        relation = row.get("relation")
        tail = row.get("tail")
        if head not in ENTITY_TYPES or tail not in ENTITY_TYPES or relation not in RELATIONS:
            return None
        if head == tail:
            return None
        return Triple(head, relation, tail, doc_id, evidence[:400], sent_id)
