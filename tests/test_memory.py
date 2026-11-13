import shutil
from pathlib import Path
import pytest
from src.utils.db_manager import PolicyDBManager

@pytest.fixture
def temp_db_dir(tmp_path):
    d = tmp_path / "test_policies"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)

def test_policy_db_lifecycle(temp_db_dir):
    team_id = "test_team_01"
    vector = [0.1] * 384

    with PolicyDBManager(team_id=team_id, db_dir=temp_db_dir) as db:
        rec_id = db.add_policy_interaction(
            project_name="CrossViT-Training",
            workload_type="vision_transformers",
            priority_tier="tier_1_critical",
            sla_deadline="under_4h",
            raw_prompt="Urgent test run for botanical benchmark",
            semantic_summary="Vision transformer training workload under critical SLA",
            vector=vector
        )
        assert rec_id is not None
        
        matches = db.get_similar_policies(query_vector=vector, limit=1)
        assert len(matches) == 1
        assert matches[0]["project_name"] == "CrossViT-Training"
        assert matches[0]["workload_type"] == "vision_transformers"
        assert matches[0]["similarity"] > 0.99
