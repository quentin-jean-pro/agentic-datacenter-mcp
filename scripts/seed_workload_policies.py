"""
Synthetic Workload Policies & SLA Seed Script.
Populates local multi-tenant DuckDB databases with realistic AI engineering workloads.
Contains ZERO proprietary or NDA data.
"""
import sys
import hashlib
from pathlib import Path
from typing import List

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.config import DATA_DIR
from src.utils.db_manager import PolicyDBManager

def generate_embedding(text: str) -> List[float]:
    """Generates a deterministic 384-dimensional normalized vector for offline seeding."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    arr = [float((b % 100) / 100.0) for b in h]
    while len(arr) < 384:
        arr.extend(arr[:min(384 - len(arr), len(arr))])
    norm = sum(x**2 for x in arr) ** 0.5
    return [x / norm for x in arr[:384]]

SYNTHETIC_WORKLOADS = [
    {
        "team_id": "cv_research_team",
        "workloads": [
            {
                "project_name": "CrossViT-Botanical-Herbarium",
                "workload_type": "vision_transformers",
                "priority_tier": "tier_1_critical",
                "sla_deadline": "under_4h",
                "raw_prompt": "Need 8x H100 GPUs for fine-tuning dual-branch multi-scale CrossViT on 50,000 herbarium sheets.",
                "semantic_summary": "High-priority distributed CrossViT vision transformer training on botanical specimen datasets."
            },
            {
                "project_name": "Botanical-Otsu-Segmentation",
                "workload_type": "batch_inference",
                "priority_tier": "tier_3_green_opportunistic",
                "sla_deadline": "overnight",
                "raw_prompt": "Run automated Otsu foreground extraction and patch rollout on historical museum scans.",
                "semantic_summary": "Flexible batch image processing job with low carbon priority to run during green energy peaks."
            }
        ]
    },
    {
        "team_id": "nlp_production_team",
        "workloads": [
            {
                "project_name": "MultiAgent-Nutrition-Llama",
                "workload_type": "llm_fine_tuning",
                "priority_tier": "tier_2_standard",
                "sla_deadline": "under_8h",
                "raw_prompt": "Fine-tuning 8B instruction model for multi-agent nutritional coaching and tool calling.",
                "semantic_summary": "Standard priority LoRA fine-tuning for domain-adapted conversational agent."
            },
            {
                "project_name": "Embedding-Index-Refresh",
                "workload_type": "batch_evaluation",
                "priority_tier": "tier_2_standard",
                "sla_deadline": "under_2h",
                "raw_prompt": "Re-index 500k nutrition facts using SentenceTransformer embeddings in DuckDB.",
                "semantic_summary": "Fast vector re-indexing pipeline with deterministic throughput."
            }
        ]
    }
]

def seed_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🌱 Seeding synthetic workload policies into '{DATA_DIR}'...")

    total_records = 0
    for team in SYNTHETIC_WORKLOADS:
        team_id = team["team_id"]
        with PolicyDBManager(team_id=team_id, db_dir=DATA_DIR) as db:
            for job in team["workloads"]:
                vec = generate_embedding(job["semantic_summary"])
                rec_id = db.add_policy_interaction(
                    project_name=job["project_name"],
                    workload_type=job["workload_type"],
                    priority_tier=job["priority_tier"],
                    sla_deadline=job["sla_deadline"],
                    raw_prompt=job["raw_prompt"],
                    semantic_summary=job["semantic_summary"],
                    vector=vec
                )
                total_records += 1
                print(f"  [Team: {team_id}] Added policy '{job['project_name']}' -> ID: {rec_id[:8]}...")

    print(f"\n✨ Successfully seeded {total_records} workload policies across {len(SYNTHETIC_WORKLOADS)} teams.")

if __name__ == "__main__":
    seed_data()

