import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


class QueryOperation(str, Enum):
    ANSWER = "ANSWER"
    SEARCH = "SEARCH"
    COUNT = "COUNT"
    LIST = "LIST"
    FILTER = "FILTER"
    SORT = "SORT"
    GROUP = "GROUP"
    FETCH = "FETCH"
    SUMMARIZE = "SUMMARIZE"
    DISPLAY = "DISPLAY"


class QueryIntent(str, Enum):
    METADATA_QUERY = "METADATA_QUERY"
    SPREADSHEET_QUERY = "SPREADSHEET_QUERY"
    SOURCE_TYPE = "SOURCE_TYPE"
    SOURCE_TYPE_QUERY = "SOURCE_TYPE_QUERY"
    FULL_CONTENT_FETCH = "FULL_CONTENT_FETCH"
    SUMMARIZATION = "SUMMARIZATION"
    IMAGE_DISPLAY = "IMAGE_DISPLAY"
    VISUAL_QA = "VISUAL_QA"
    IMAGE_FACE_QUERY = "IMAGE_FACE_QUERY"
    FACE_OPERATIONS = "FACE_OPERATIONS"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    CORRECTION = "CORRECTION"

class FaceIntent(str, Enum):
    NONE = "none"
    COUNT = "face_count"
    IDENTIFICATION = "face_identification"
    PRESENCE = "face_presence"
    SEARCH = "face_search"


class RequestScope(str, Enum):
    """
    Defines the structural scope of information requested by the user.
    """
    COMPLETE_FILE = "COMPLETE_FILE"
    SUMMARY = "SUMMARY"
    QUESTION_ANSWER = "QUESTION_ANSWER"
    METADATA_ONLY = "METADATA_ONLY"


class Modality(str, Enum):
    IMAGE = "image"
    DOCUMENT = "document"
    PDF = "pdf"
    DOCX = "docx"
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
    candidate_sources: List[str] = field(default_factory=list)

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
    operation: QueryOperation
    scope: RequestScope
    modality: Modality
    file_type: Optional[str]
    face_intent: FaceIntent = FaceIntent.NONE
    face_person_name: Optional[str] = None
    source_spec: SourceSpec = field(default_factory=SourceSpec)
    filters: QueryFilters = field(default_factory=QueryFilters)
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
            "operation": self.operation.value,
            "request_scope": self.scope.value.lower(),
            "temporal_intent": "TEMPORAL_FILE_QUERY" if self.intent == QueryIntent.METADATA_QUERY else "none",
            "face_intent": self.face_intent.value,
            "face_person_name": self.face_person_name,
            "is_visual_qa": self.intent == QueryIntent.VISUAL_QA,
            "is_ocr_query": self.metadata.get("is_ocr_query", False),
            "is_image_summary_query": self.intent == QueryIntent.SUMMARIZATION and self.modality == Modality.IMAGE,
            "modality": self.modality.value,
            "file_type": self.file_type,
            "source_hint": self.source_spec.source_hint,
            "canonical_source_id": self.source_spec.canonical_path,
            "unresolved_explicit_source": self.source_spec.is_explicit and not self.source_spec.is_resolved,
            "source_confidence": self.source_spec.confidence,
            "candidate_sources": self.source_spec.candidate_sources,
            "target_language": self.filters.target_language,
            "is_correction": self.is_correction,
            "start_datetime": self.filters.start_datetime,
            "end_datetime": self.filters.end_datetime,
            "date_label": self.filters.date_label,
            "extracted_limit": self.filters.extracted_limit,
            "needs_database": True
        }
