from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv
import os

load_dotenv()


class LLMProviderSettings(BaseSettings):
    temperature: float = 0.8  # Increase for more diversity
    max_tokens: int | None = None
    max_retries: int = 3


# OpenAI settings for embeddings
class OpenAIEmbeddingSettings(LLMProviderSettings):
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    default_model: str = "text-embedding-ada-002"


# OpenAI settings for summaries
class OpenAISummarySettings(LLMProviderSettings):
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    default_model: str = "gpt-3.5-turbo"


# Hugging Face settings for embeddings
class HuggingFaceEmbeddingSettings(LLMProviderSettings):
    default_model: str = "sentence-transformers/all-MiniLM-L6-v2"


# Hugging Face settings for summaries
class HuggingFaceSummarySettings(LLMProviderSettings):
    default_model: str = "sshleifer/distilbart-cnn-12-6"  # Default summarization model
    # dynamic_length: bool = True  # Default to true as per config.yml


class Settings(BaseSettings):
    app_name: str = "Embedding and Summary Generator"
    openai_embeddings: OpenAIEmbeddingSettings = OpenAIEmbeddingSettings()
    openai_summaries: OpenAISummarySettings = OpenAISummarySettings()
    huggingface_embeddings: HuggingFaceEmbeddingSettings = (
        HuggingFaceEmbeddingSettings()
    )
    huggingface_summaries: HuggingFaceSummarySettings = HuggingFaceSummarySettings()


@lru_cache
def get_settings():
    return Settings()
