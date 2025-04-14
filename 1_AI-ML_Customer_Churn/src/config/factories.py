from typing import Any, List
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from src.config.settings import get_settings


class EmbeddingFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), f"{provider}_embeddings")
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        if self.provider == "openai":
            if not self.settings.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            return OpenAI(api_key=self.settings.api_key)
        elif self.provider == "huggingface":
            return SentenceTransformer(self.settings.default_model)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def get_embeddings(self, text: str) -> List[float]:
        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=[text], model=self.settings.default_model
            )
            return response.data[0].embedding
        elif self.provider == "huggingface":
            return self.client.encode(text).tolist()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")


class SummaryFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), f"{provider}_summaries")
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        if self.provider == "openai":
            if not self.settings.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            return OpenAI(api_key=self.settings.api_key)
        elif self.provider == "huggingface":
            return pipeline("summarization", model=self.settings.default_model)
        else:
            raise ValueError(f"Unsupported summary provider: {self.provider}")

    def summarize(self, text: str) -> str:
        if self.provider == "openai":
            prompt = f"Summarize the following customer ticket focusing on the main complaint or request:\n\n{text}\n\nSummary:"
            response = self.client.chat.completions.create(
                model=self.settings.default_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.temperature,
            )
            return response.choices[0].message.content.strip()
        elif self.provider == "huggingface":
            input_length = len(text.split())
            max_length = (
                min(
                    self.settings.max_tokens if self.settings.max_tokens else 20,
                    max(2, input_length // 5),  # e.g., 2 for 9 tokens
                )
                # if getattr(self.settings, "dynamic_length", True)
                # else self.settings.max_tokens
            )
            min_length = max(1, input_length // 6)  # e.g., 1 for 9 tokens
            # Ensure temperature is positive
            temperature = (
                self.settings.temperature if self.settings.temperature > 0 else 1.0
            )
            if self.settings.temperature <= 0:
                raise ("Temperature <= 0 detected; defaulting to 1.0 for sampling.")
            summary = self.client(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,  # Enable sampling for diversity
                temperature=self.settings.temperature,
                top_k=50,
                top_p=0.95,
            )
            return summary[0]["summary_text"]
        else:
            raise ValueError(f"Unsupported summary provider: {self.provider}")
