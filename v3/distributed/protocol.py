"""Shared data models for server/client communication."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class RegisterRequest(BaseModel):
    hostname: str
    gpu_info: str


class RegisterResponse(BaseModel):
    client_id: str


class StatusPush(BaseModel):
    state: str                          # idle | syncing | running | uploading | error
    job_id: Optional[str] = None
    run_name: Optional[str] = None
    rep: Optional[int] = None
    fold: Optional[int] = None
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    last_epoch_secs: Optional[float] = None
    last_auc: Optional[float] = None
    error: Optional[str] = None


class ClientInfo(BaseModel):
    client_id: str
    hostname: str
    gpu_info: str
    status: StatusPush
    last_seen: str


class JobSpec(BaseModel):
    job_id: str
    run_name: str
    module: str                         # e.g. "v3.scripts.main.run_cv"
    args: list[str]
    output_dir: str = "v3/results"      # local subdir for rsync-back


class PollResponse(BaseModel):
    job: Optional[JobSpec] = None
    please_reregister: bool = False


class JobResult(BaseModel):
    job_id: str
    success: bool
    error_msg: Optional[str] = None


class JobSubmit(BaseModel):
    run_name: str
    module: str = "v3.scripts.main.run_cv"
    args: list[str]
    output_dir: str = "v3/results"
    priority: int = 0
