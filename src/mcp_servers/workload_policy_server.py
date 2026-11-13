import os
import sys
import logging
import signal
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError:
        class FastMCP:
            def __init__(self, name: str, host: str = "0.0.0.0", port: int = 8000):
                self.name = name
                self.host = host
                self.port = port
            def tool(self):
                def decorator(fn):
                    return fn
                return decorator
            def run(self, transport="sse"):
                print(f"Mock FastMCP '{self.name}' running on {self.host}:{self.port} ({transport})")

# Add parent directory to sys.path if run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.config import POLICY_MCP_HOST, POLICY_MCP_PORT, DATA_DIR, EMBEDDING_MODEL_NAME
from src.utils.db_manager import PolicyDBManager

# --- 1. SETUP & LOGGING ---
logger = logging.getLogger("DataCenter_Policy_MCP")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = RotatingFileHandler("datacenter_policy.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --- 2. VECTOR EMBEDDING MODEL INITIALIZATION ---
logger.info(f"Loading local SentenceTransformer model: '{EMBEDDING_MODEL_NAME}'...")
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("✅ SentenceTransformer loaded successfully.")
except Exception as e:
    logger.warning(f"⚠️ sentence-transformers not available or offline: {e}. Falling back to deterministic pseudo-embedding.")
    import hashlib
    class FallbackEmbedding:
        def encode(self, text: str) -> List[float]:
            # Deterministic 384-dimensional normalized vector for offline testing
            h = hashlib.sha256(text.encode('utf-8')).digest()
            arr = [float((b % 100) / 100.0) for b in h]
            while len(arr) < 384:
                arr.extend(arr[:min(384 - len(arr), len(arr))])
            norm = sum(x**2 for x in arr) ** 0.5
            return [x / norm for x in arr[:384]]
    embedding_model = FallbackEmbedding()

# --- 3. FASTMCP SERVER INITIALIZATION ---
mcp = FastMCP("DataCenter_Policy_Server", host=POLICY_MCP_HOST, port=POLICY_MCP_PORT)

# --- 4. EXPOSED MCP TOOLS ---

@mcp.tool()
async def ping_policy_server() -> str:
    """Verifies health of the workload policy & vector database server."""
    return "SUCCESS: DataCenter Workload Policy Server is operational."

@mcp.tool()
async def add_workload_policy_interaction(
    team_id: str,
    project_name: str,
    workload_type: str,
    priority_tier: str,
    sla_deadline: str,
    raw_prompt: str,
    semantic_summary: str
) -> str:
    """
    Persists structured AI team scheduling request and generates semantic vector embeddings.
    Multi-tenant isolation: stored in an independent DuckDB database file per engineering team.
    """
    logger.info(f"💾 Saving workload policy interaction for team '{team_id}', project '{project_name}'...")
    try:
        vec = embedding_model.encode(semantic_summary)
        if hasattr(vec, 'tolist'):
            vec = vec.tolist()
        vec = [float(x) for x in vec]

        with PolicyDBManager(team_id=team_id, db_dir=DATA_DIR) as db:
            record_id = db.add_policy_interaction(
                project_name=project_name,
                workload_type=workload_type,
                priority_tier=priority_tier,
                sla_deadline=sla_deadline,
                raw_prompt=raw_prompt,
                semantic_summary=semantic_summary,
                vector=vec
            )

        logger.info(f"✅ Record '{record_id}' stored successfully in DuckDB.")
        return f"SUCCESS: Workload policy recorded with ID {record_id} for team '{team_id}'."
    except Exception as e:
        logger.exception(f"❌ Failed to persist policy interaction: {e}")
        return f"ERROR: Could not persist policy interaction: {str(e)}"

@mcp.tool()
async def get_team_context(team_id: str, query: str, limit: int = 3) -> str:
    """
    Queries past workload requests, SLA contracts, and scheduling decisions for a team using vector cosine similarity.
    """
    logger.info(f"🔍 Searching past context for team '{team_id}' with query: '{query}'...")
    try:
        q_vec = embedding_model.encode(query)
        if hasattr(q_vec, 'tolist'):
            q_vec = q_vec.tolist()
        q_vec = [float(x) for x in q_vec]

        with PolicyDBManager(team_id=team_id, db_dir=DATA_DIR) as db:
            matches = db.get_similar_policies(query_vector=q_vec, limit=limit)

        if not matches:
            return f"INFO: No previous workload history found for team '{team_id}'."

        formatted_results = []
        for idx, match in enumerate(matches, 1):
            formatted_results.append(
                f"[{idx}] Project: {match['project_name']} | Type: {match['workload_type']} "
                f"| Priority: {match['priority_tier']} | SLA: {match['sla_deadline']}\n"
                f"    Summary: {match['semantic_summary']} (Similarity: {match['similarity']:.2f})"
            )

        return "\n".join(formatted_results)
    except Exception as e:
        logger.exception(f"❌ Error during vector retrieval: {e}")
        return f"ERROR: Vector retrieval failed: {str(e)}"

if __name__ == "__main__":
    logger.info(f"Starting DataCenter Policy MCP Server on {POLICY_MCP_HOST}:{POLICY_MCP_PORT} (SSE Transport)...")
    mcp.run(transport="sse")
