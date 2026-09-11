import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


class QueryIntent(str, Enum):
    """
    Core semantic intents supported by the RAG architecture.
    """
    METADATA_QUERY = "METADATA_QUERY"
    FULL_CONTENT_FETCH = "FULL_CONTENT_FETCH"
    SUMMARIZATION = "SUMMARIZATION"
    SPEAKER_ANALYSIS = "SPEAKER_ANALYSIS"
    VISUAL_QA = "VISUAL_QA"
    FACE_OPERATIONS = "FACE_OPERATIONS"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    CORRECTION = "CORRECTION"


class RequestScope(str, Enum):
    """
    Defines the structural scope of information requested by the user.
    """
    COMPLETE_FILE = "COMPLETE_FILE"
    SUMMARY = "SUMMARY"
    QUESTION_ANSWER = "QUESTION_ANSWER"
    METADATA_ONLY = "METADATA_ONLY"


class Modality(str, Enum):
    """
    Target media or document modality.
    """
    AUDIO = "audio"
    IMAGE = "image"
    DOCUMENT = "document"
    PDF = "pdf"
    DOCX = "docx"
    VIDEO = "video"
    ALL = "all"


@dataclass
class SourceSpec:
    """
    Represents the source resolution status for a query.
    """
    source_hint: Optional[str] = None
    canonical_path: Optional[str] = None
    is_explicit: bool = False
    is_resolved: bool = False
    is_ambiguous: bool = False
    confidence: float = 0.0

    @property
    def filename(self) -> Optional[str]:
        if self.canonical_path:
            return os.path.basename(self.canonical_path)
        if self.source_hint and not self.source_hint.startswith("UNRESOLVED_"):
            return os.path.basename(self.source_hint)
        return None


@dataclass
class QueryFilters:
    """
    Temporal, limit, and language constraints extracted from the user query.
    """
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    date_label: Optional[str] = None
    extracted_limit: Optional[int] = None
    target_language: Optional[str] = None


@dataclass
class QueryPlan:
    """
    The stable internal contract produced by the Query Understanding layer
    and consumed by the Intent Router.
    """
    raw_query: str
    normalized_query: str
    intent: QueryIntent
    scope: RequestScope
    modality: Modality
    source_spec: SourceSpec
    filters: QueryFilters
    is_correction: bool = False
    correction_details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts QueryPlan into a backward-compatible dictionary format
        used by existing legacy subsystems.
        """
        return {
            "intent": self.intent.value,
            "request_scope": self.scope.value.lower(),
            "temporal_intent": "TEMPORAL_FILE_QUERY" if self.intent == QueryIntent.METADATA_QUERY else "none",
            "face_intent": "none",
            "is_visual_qa": self.intent == QueryIntent.VISUAL_QA,
            "is_ocr_query": self.metadata.get("is_ocr_query", False),
            "is_image_summary_query": self.intent == QueryIntent.SUMMARIZATION and self.modality == Modality.IMAGE,
            "modality": self.modality.value,
            "source_hint": self.source_spec.source_hint,
            "canonical_source_id": self.source_spec.canonical_path,
            "unresolved_explicit_source": self.source_spec.is_explicit and not self.source_spec.is_resolved,
            "source_confidence": self.source_spec.confidence,
            "target_language": self.filters.target_language,
            "is_correction": self.is_correction,
            "start_datetime": self.filters.start_datetime,
            "end_datetime": self.filters.end_datetime,
            "date_label": self.filters.date_label,
            "extracted_limit": self.filters.extracted_limit,
            "needs_database": True
        }
