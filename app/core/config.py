from pydantic_settings import BaseSettings
from functools import lru_cache

import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

from pydantic import field_validator
import json

class Settings(BaseSettings):
    # App
    app_name: str = "TalentAgent"
    auth_mode: str = "dev"  # dev | puter
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:5174", "*"]
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [x.strip() for x in v.split(",") if x.strip()]
        return v
    
    # LLM Provider: 'groq' | 'openai' | 'anthropic'
    llm_provider: str = "groq"
    
    # Groq LLM
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    
    # OpenAI LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    
    # Anthropic LLM
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Hugging Face
    hf_token: str = ""
    
    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimensions: int = 1024
    
    # Database
    database_url: str
    
    # Supabase
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    
    # GitHub
    github_token: str

    # Rate Limiting
    groq_rpm_limit: int = 20
    github_rpm_limit: int = 20
    github_rph_limit: int = 850
    
    class Config:
        env_file = str(_ENV_PATH)
        env_file_encoding = "utf-8"
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
