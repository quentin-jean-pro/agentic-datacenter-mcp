import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent

# Operating Mode: DRY_RUN mocks real cluster actuation and telemetry sink
DRY_RUN: bool = os.getenv("DRY_RUN", "True").lower() == "true"

# LLM API Keys (optional; system includes offline fallback mock)
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Networking Configuration for FastMCP Servers
COMPUTE_MCP_HOST: str = os.getenv("COMPUTE_MCP_HOST", "0.0.0.0")
COMPUTE_MCP_PORT: int = int(os.getenv("COMPUTE_MCP_PORT", "8001"))
COMPUTE_MCP_URL: str = os.getenv("COMPUTE_MCP_URL", f"http://localhost:{COMPUTE_MCP_PORT}/sse")

POLICY_MCP_HOST: str = os.getenv("POLICY_MCP_HOST", "0.0.0.0")
POLICY_MCP_PORT: int = int(os.getenv("POLICY_MCP_PORT", "8002"))
POLICY_MCP_URL: str = os.getenv("POLICY_MCP_URL", f"http://localhost:{POLICY_MCP_PORT}/sse")

# Cluster Telemetry Sinks
TELEMETRY_ENDPOINT: str = os.getenv("TELEMETRY_ENDPOINT", "http://localhost:9090/api/v1")
PROMETHEUS_JOB: str = os.getenv("PROMETHEUS_JOB", "datacenter_gpu_metrics")

# Storage & Semantic Models
DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data_policies")
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
