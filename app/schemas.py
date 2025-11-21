# ==================== app/schemas.py ====================
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class EmbedRequest(BaseModel):
    resource: str = Field(..., min_length=1, max_length=100, description="Resource identifier")
    timestamp: Optional[str] = Field(None, description="ISO format timestamp")
    severity: Optional[str] = Field(None, pattern="^(low|medium|high|critical)$")
    logs: str = Field(..., min_length=1, description="Log content to embed")
    
    @field_validator('logs')
    @classmethod
    def validate_logs(cls, v):
        if not v or not v.strip():
            raise ValueError("logs cannot be empty or whitespace")
        return v.strip()

class EmbedResponse(BaseModel):
    status: str
    indexed_documents: int
    document_ids: List[str]

class AnalyzeRequest(BaseModel):
    resource: Optional[str] = Field(None, max_length=100)
    timestamp: Optional[str] = None
    logs: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(3, ge=1, le=10)
    
    @field_validator('logs')
    @classmethod
    def validate_logs(cls, v):
        if not v or not v.strip():
            raise ValueError("logs cannot be empty or whitespace")
        return v.strip()

class SimilarIncident(BaseModel):
    id: str
    content: str
    timestamp: str
    resource: str
    severity: Optional[str]
    score: float

class AnalyzeResponse(BaseModel):
    summary: str
    probable_root_cause: str
    recommended_action: str
    similar_incidents: List[SimilarIncident]
    confidence: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str