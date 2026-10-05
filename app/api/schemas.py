"""Request and response models for the API. Code guide: section "API" (sec:api)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Subscriber(BaseModel):
    customer_id: int
    tenure_months: int = Field(ge=0)
    plan_annual: int = Field(ge=0, le=1)
    sessions_30d: int = Field(ge=0)
    days_since_login: int = Field(ge=0)
    price_sensitivity: float = Field(ge=0, le=1)
    payment_failures_90d: int = Field(ge=0)
    support_tickets_90d: int = Field(ge=0)
    monthly_fee: float = Field(gt=0)
    region: str
    value: float = Field(gt=0)
    offer_cost: float = Field(ge=0)


class ScoreRequest(BaseModel):
    subscribers: list[Subscriber]


class ScoreRow(BaseModel):
    customer_id: int
    tau_hat: float
    churn_risk: float
    net_value: float
    recommend_offer: bool


class ScoreResponse(BaseModel):
    model_version: str
    scores: list[ScoreRow]


class RecommendRequest(BaseModel):
    budget: int = Field(default=2000, gt=0, le=50_000)


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    budget: int | None = Field(default=None, gt=0, le=50_000)


class AgentResponse(BaseModel):
    intent: str
    narrative: str
    review: dict[str, Any] | None = None
    policy_summary: dict[str, Any] | None = None
    campaign_id: str | None = None
    trace: list[str]
    provider: str
    model_version: str | None = None


class ApproveRequest(BaseModel):
    reviewer: str = Field(min_length=1)
    approve: bool
    exclude_ids: list[int] = []
    note: str = ""
