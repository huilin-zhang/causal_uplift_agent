"""Human-authored policy targets and evidence-backed campaign designs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PolicyTargets(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    objective: Literal["incremental_retention", "net_value"] = "incremental_retention"
    max_contacts: int = Field(default=2000, ge=0, le=50_000)
    total_budget: float | None = Field(default=None, ge=0)
    min_roi: float = Field(default=0, ge=0)  # incremental gross value / expected cost
    min_new_user_share: float = Field(default=0, ge=0, le=1)
    holdout_fraction: float = Field(default=0.1, gt=0, lt=1)
    max_contacts_per_week: int = Field(default=2, ge=1)
    approval_ttl_hours: int = Field(default=48, ge=1, le=168)
    automation_level: Literal["assisted"] = "assisted"


class CampaignDesign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    audience: Literal["all", "monthly"] = "all"
    intervention: Literal["retention_discount"] = "retention_discount"
    # The existing RCT estimates the complete intervention, not channel effects.
    channel: Literal["existing"] = "existing"
    message_type: Literal["retention"] = "retention"
    tone: Literal["neutral"] = "neutral"
    timing: Literal["existing"] = "existing"
    length: Literal["short"] = "short"
