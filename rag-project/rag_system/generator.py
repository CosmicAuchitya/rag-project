"""Answer generation utilities for local and remote LLM providers."""

from __future__ import annotations

from .config import LLMConfig
from .models import AnswerResult, RetrievalResult


class AnswerGenerator:
    """Generates answers from retrieved context using a configurable provider."""

    def __init__(self, config: LLMConfig) -> None:
        """Stores configuration and initializes the client when needed."""

        self.config = config
        self.client = None
        self.tokenizer = None
        self.model = None
        self.model_device = "cpu"

        if config.provider in {"openai", "openai_compatible"}:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("OpenAI-based generation requires the 'openai' package.") from exc

            self.client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url)
        elif config.provider == "huggingface_local":
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
            except ImportError as exc:
                raise ImportError(
                    "Local Hugging Face generation requires the 'transformers' package."
                ) from exc

            if config.hf_task == "text2text-generation":
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
                self.model_device = "cuda" if config.device >= 0 and torch.cuda.is_available() else "cpu"
                self.model.to(self.model_device)
            else:
                self.client = pipeline(
                    task=config.hf_task,
                    model=config.model_name,
                    device=config.device,
                )

    def generate_answer(self, query: str, retrieved_results: list[RetrievalResult]) -> AnswerResult:
        """Builds a grounded prompt and generates an answer from retrieved context."""

        prompt = self._build_prompt(query=query, retrieved_results=retrieved_results)

        if self.config.provider == "extractive":
            answer = self._generate_extractive_answer(query=query, retrieved_results=retrieved_results)
        elif self.config.provider in {"openai", "openai_compatible"}:
            answer = self._generate_openai_answer(prompt=prompt)
        elif self.config.provider == "huggingface_local":
            answer = self._generate_huggingface_answer(prompt=prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

        return AnswerResult(
            query=query,
            answer=answer,
            retrieved_results=retrieved_results,
            prompt=prompt,
        )

    def _build_prompt(self, query: str, retrieved_results: list[RetrievalResult]) -> str:
        """Formats retrieved chunks into a grounded RAG prompt."""

        context_blocks: list[str] = []

        for index, result in enumerate(retrieved_results[: self.config.max_context_results], start=1):
            source = result.metadata.get("source", "unknown")
            context_blocks.append(
                f"[Context {index}] score={result.score:.4f} source={source}\n{result.text}"
            )

        context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."

        return (
            "You are a precise question-answering assistant.\n"
            "Answer the user using only the supplied context.\n"
            "Write a concise answer in 2 to 4 complete sentences.\n"
            "If the answer is not present, say that the context does not contain enough information.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

    def _generate_openai_answer(self, prompt: str) -> str:
        """Generates an answer using an OpenAI-compatible chat completion API."""

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": "You answer questions strictly from retrieved context."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def _generate_extractive_answer(self, query: str, retrieved_results: list[RetrievalResult]) -> str:
        """Provides a deterministic fallback answer when no LLM is configured."""

        if not retrieved_results:
            return "The retrieved context does not contain enough information to answer the question."

        best_result = retrieved_results[0]
        source = best_result.metadata.get("source", "unknown")
        return (
            f"Top retrieved context from {source}: {best_result.text}\n\n"
            f"Question: {query}\n"
            "This is an extractive fallback. Configure an OpenAI-compatible LLM for generative answers."
        )

    def _generate_huggingface_answer(self, prompt: str) -> str:
        """Generates an answer with a local Hugging Face pipeline."""

        if self.model is not None and self.tokenizer is not None:
            return self._generate_huggingface_seq2seq_answer(prompt)

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0.0,
            "truncation": True,
        }

        if self.config.temperature > 0.0:
            generation_kwargs["temperature"] = self.config.temperature

        response = self.client(prompt, **generation_kwargs)

        if not response:
            return "The local model returned an empty response."

        first_item = response[0]
        return first_item.get("generated_text") or first_item.get("summary_text") or str(first_item)

    def _generate_huggingface_seq2seq_answer(self, prompt: str) -> str:
        """Generates an answer with a locally loaded seq2seq model."""

        try:
            import torch
        except ImportError as exc:
            raise ImportError("Local seq2seq generation requires the 'torch' package.") from exc

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        encoded = {key: value.to(self.model_device) for key, value in encoded.items()}
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0.0,
        }

        if self.config.temperature > 0.0:
            generation_kwargs["temperature"] = self.config.temperature

        with torch.no_grad():
            output_tokens = self.model.generate(
                **encoded,
                **generation_kwargs,
            )

        return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
