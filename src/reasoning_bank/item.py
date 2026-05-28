"""MemoryItem — single unit stored in the ReasoningBank."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


MemorySource = Literal["success", "failure", "matts_contrast", "manual"]


class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:10]}")
    title: str
    description: str
    content: str
    source: MemorySource
    task_signature: str
    scope: str = "global"
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: datetime | None = None
    use_count: int = 0
    confidence: float = 1.0
    related_ids: list[str] = Field(default_factory=list)

    def signature_text(self) -> str:
        return f"{self.title}\n{self.description}\n{self.content}"
