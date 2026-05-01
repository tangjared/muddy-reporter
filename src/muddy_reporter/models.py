from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


Confidence = Literal["low", "medium", "high"]
FindingLabel = Literal["fact", "inference", "question", "speculation"]
FindingCategory = Literal[
    "accounting",
    "governance",
    "disclosure",
    "operations",
    "capital_structure",
    "related_parties",
    "regulatory_legal",
    "other",
]


class SourceDoc(BaseModel):
    doc_id: str
    ticker: str
    cik: str
    filing_type: str
    filing_date: date | None = None
    accession: str | None = None
    primary_url: str
    local_path: str
    sha256: str


class Citation(BaseModel):
    doc_id: str
    url: str
    excerpt: str = Field(..., description="Quoted excerpt supporting the point.")


class Finding(BaseModel):
    title: str
    category: FindingCategory
    label: FindingLabel
    confidence: Confidence
    claim_or_observation: str
    why_it_matters: str
    counterpoints_or_alt_explanations: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class Report(BaseModel):
    ticker: str
    company_name: str | None = None
    generated_at_iso: str
    snapshot: dict
    core_thesis: str
    red_flags: list[Finding]
    management_claims_vs_counterpoints: list[dict]
    concerns_by_category: dict
    open_questions: list[str]
    conclusion: str
    limitations: list[str]
    financial_anomalies: list[dict] = Field(default_factory=list)
    financial_table: list[dict] = Field(default_factory=list)
    forensic_scores: dict = Field(default_factory=dict)
    risk_grade: dict = Field(default_factory=dict)
    provider_info: dict = Field(default_factory=dict)


class PipelineResult(BaseModel):
    out_html_path: str
    out_json_path: str
    sources: list[SourceDoc]
    report: Report

