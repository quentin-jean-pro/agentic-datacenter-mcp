import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

logger = logging.getLogger("DataCenter.DBManager")


class PolicyDBManager:
    """
    DuckDB Manager for isolated multi-tenant SLA & Workload policy storage.
    Each engineering team gets a dedicated database file in DATA_DIR to guarantee data isolation.
    Operates as a Context Manager to ensure immediate connection cleanup and zero database locks.
    """

    def __init__(self, team_id: str, db_dir: Path | str | None = None):
        self.team_id = team_id
        safe_team_id = "".join([c for c in team_id if c.isalnum() or c in ("_", "-")]).strip()
        if not safe_team_id:
            safe_team_id = "default_team"

        self.db_dir = Path(db_dir) if db_dir else Path("data_policies")
        self.db_path = self.db_dir / f"team_{safe_team_id}.db"
        self.conn: duckdb.DuckDBPyConnection | None = None

    def __enter__(self):
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_db()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            try:
                self.conn.close()
                logger.debug(f"DuckDB connection safely closed for team '{self.team_id}'.")
            except Exception as e:
                logger.error(f"Error while closing DuckDB connection: {e}")
            self.conn = None

    def _init_db(self):
        """Initializes the vector workload policies table schema."""
        try:
            self.conn.execute("SELECT workload_type FROM workload_policies LIMIT 1")
        except Exception:
            logger.info("Initializing workload_policies table schema...")
            self.conn.execute("DROP TABLE IF EXISTS workload_policies;")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS workload_policies (
                id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                project_name VARCHAR,
                workload_type VARCHAR,
                priority_tier VARCHAR,
                sla_deadline VARCHAR,
                raw_prompt VARCHAR,
                semantic_summary VARCHAR,
                vector FLOAT[]
            );
        """)

    def add_policy_interaction(
        self,
        project_name: str,
        workload_type: str,
        priority_tier: str,
        sla_deadline: str,
        raw_prompt: str,
        semantic_summary: str,
        vector: list[float],
    ) -> str:
        """
        Inserts a new workload scheduling intent with normalized embedding vector.
        """
        interaction_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        query = """
            INSERT INTO workload_policies (
                id, timestamp, project_name, workload_type, priority_tier,
                sla_deadline, raw_prompt, semantic_summary, vector
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            query,
            (
                interaction_id,
                now,
                project_name,
                workload_type,
                priority_tier,
                sla_deadline,
                raw_prompt,
                semantic_summary,
                vector,
            ),
        )
        logger.info(f"Policy interaction '{interaction_id}' stored for team '{self.team_id}'.")
        return interaction_id

    def get_similar_policies(
        self, query_vector: list[float], limit: int = 3
    ) -> list[dict[str, Any]]:
        """
        Performs vector cosine similarity search over stored team workload policies.
        Uses in-memory NumPy operations for high throughput with zero external vector DB dependencies.
        """
        cursor = self.conn.execute("""
            SELECT id, timestamp, project_name, workload_type, priority_tier,
                   sla_deadline, raw_prompt, semantic_summary, vector
            FROM workload_policies
        """)
        rows = cursor.fetchall()

        if not rows:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-9:
            return []

        results = []
        vectors = []
        metadata = []

        for row in rows:
            vectors.append(row[8])
            metadata.append(
                {
                    "id": row[0],
                    "timestamp": str(row[1]),
                    "project_name": row[2],
                    "workload_type": row[3],
                    "priority_tier": row[4],
                    "sla_deadline": row[5],
                    "raw_prompt": row[6],
                    "semantic_summary": row[7],
                }
            )

        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1e-9

        # Cosine similarity: (A · B) / (||A|| * ||B||)
        similarities = np.dot(matrix, q_vec) / (norms * q_norm)

        top_indices = np.argsort(similarities)[::-1][:limit]

        for idx in top_indices:
            item = metadata[idx]
            item["similarity"] = float(similarities[idx])
            results.append(item)

        return results
