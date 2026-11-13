import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.mcp_servers.compute_telemetry_server import build_mcp_response
from src.utils.db_manager import PolicyDBManager
from tests.test_workload_intentions import test_pydantic_structured_analysis_valid, test_intentions_coverage

def test_intentions():
    print("Testing Workload Intentions & Pydantic Schema...")
    test_pydantic_structured_analysis_valid()
    test_intentions_coverage()
    print("  -> test_intentions: PASSED ✅")

def test_contracts():
    print("Testing JSON Contracts & Envelope...")
    raw = build_mcp_response("SUCCESS", {"node_id": "node-alpha", "power_cap_watts": 350}, "node_power_cap", {"cluster": "eu_west"}, {"power": 350})
    data = json.loads(raw)
    assert data["execution_status"] == "SUCCESS", "Status must be SUCCESS"
    assert data["system_context"]["node_id"] == "node-alpha", "Node ID must match"
    assert data["db_trace"]["measurement"] == "node_power_cap"
    print("  -> test_contracts: PASSED ✅")

def test_memory():
    print("Testing DuckDB Vector Memory & Cosine Similarity...")
    tmp = tempfile.mkdtemp()
    try:
        dummy_vector = [0.1] * 384
        with PolicyDBManager("test_engineer_team", db_dir=tmp) as db:
            rec_id = db.add_policy_interaction(
                project_name="CrossViT-Training",
                workload_type="vision_transformers",
                priority_tier="tier_1_critical",
                sla_deadline="under_4h",
                raw_prompt="Urgent test run for botanical benchmark",
                semantic_summary="Vision transformer training workload under critical SLA",
                vector=dummy_vector
            )
            assert rec_id is not None, "Record ID must be generated"
            
            results = db.get_similar_policies(dummy_vector, limit=1)
            assert len(results) == 1, "Must return 1 match"
            assert results[0]["project_name"] == "CrossViT-Training"
            assert results[0]["similarity"] > 0.99, "Cosine similarity must be ~1.0"
        print("  -> test_memory: PASSED ✅")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    test_intentions()
    test_contracts()
    test_memory()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
