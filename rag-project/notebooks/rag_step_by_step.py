# %% [markdown]
# # Production RAG System Walkthrough
#
# This file is organized with `# %%` notebook cells so it can be run step-by-step.

# %%
from pathlib import Path
import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
LOCAL_SITE_PACKAGES = PROJECT_DIR / "vendor_pkgs"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if LOCAL_SITE_PACKAGES.exists() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if load_dotenv is not None:
    load_dotenv()

from rag_system import RAGPipeline, build_default_config
from rag_system.models import EvaluationExample

# %%
# Step 1: Build a shared configuration.
config = build_default_config(PROJECT_DIR)

config.collection_name = "sample_docs"
config.debug = True
config.chunking.chunk_size = 220
config.chunking.chunk_overlap = 40
config.retrieval.top_k = 3
config.retrieval.min_score = 0.20

# Make the providers configurable from environment variables.
config.embeddings.provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
config.embeddings.model_name = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
config.embeddings.openai_api_key = os.getenv("OPENAI_API_KEY")
config.embeddings.openai_base_url = os.getenv("OPENAI_BASE_URL")

config.llm.provider = os.getenv("LLM_PROVIDER", "extractive")
config.llm.model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
config.llm.openai_api_key = os.getenv("OPENAI_API_KEY")
config.llm.openai_base_url = os.getenv("OPENAI_BASE_URL")

pipeline = RAGPipeline(config=config)

# %%
# Step 2: Load data, clean it, chunk it, embed it, and build the FAISS index.
sample_data_dir = PROJECT_DIR / "data" / "sample"
index_stats = pipeline.build_index([sample_data_dir])
print("Index statistics:", index_stats)

# %%
# Step 3: Ask a dynamic query.
query = "Which plan includes audit logs and what is its monthly price?"
answer = pipeline.answer_query(query=query, top_k=3, min_score=0.20)

print("\nFinal answer:\n")
print(answer.answer)

print("\nRetrieved context:\n")
for item in answer.retrieved_results:
    print(
        {
            "score": round(item.score, 4),
            "source": item.metadata.get("source"),
            "row_number": item.metadata.get("row_number"),
        }
    )

# %%
# Step 4: Evaluate retrieval quality.
evaluation_examples = [
    EvaluationExample(
        query="How long do new hires have to complete security training?",
        expected_sources=["company_handbook.txt"],
        expected_keywords=["14 days", "security training"],
    ),
    EvaluationExample(
        query="Which plan has anomaly alerts?",
        expected_sources=["product_faq.txt", "pricing.csv"],
        expected_keywords=["anomaly alerts", "Pro"],
    ),
]

metrics = pipeline.evaluate(examples=evaluation_examples, top_k=3)
print("\nRetrieval evaluation:\n")
print(metrics)

# %%
# Step 5: Example output expectation.
#
# If the required packages are installed, the query above should retrieve the
# Enterprise pricing row and the product FAQ chunk mentioning audit logs.
# With a generative LLM enabled, a typical answer would say:
# "The Enterprise plan includes audit logs and costs 999 USD per month."
