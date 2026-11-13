try:
    import pytest
except ImportError:
    pass
from pydantic import ValidationError
from src.agents.schemas import WorkloadRequestAnalysis

def test_pydantic_structured_analysis_valid():
    """Verifies that structured analysis correctly parses valid AI workload requirements."""
    data = {
        "project_name": "CrossViT-Botanical",
        "workload_type": "vision_transformers",
        "gpus_requested": 8,
        "priority_tier": "tier_1_critical",
        "sla_deadline": "under_4h",
        "semantic_summary": "Urgent allocation of 8 GPUs for CrossViT classification benchmark."
    }
    model = WorkloadRequestAnalysis(**data)
    assert model.project_name == "CrossViT-Botanical"
    assert model.gpus_requested == 8
    assert model.priority_tier == "tier_1_critical"

def test_pydantic_structured_analysis_defaults():
    """Verifies default values and field validation."""
    data = {
        "project_name": "NLP-Llama",
        "workload_type": "llm_training",
        "priority_tier": "tier_2_standard",
        "sla_deadline": "under_8h",
        "semantic_summary": "Fine-tuning prompt coaching model."
    }
    model = WorkloadRequestAnalysis(**data)
    assert model.gpus_requested == 4  # Default value
    assert model.sla_deadline == "under_8h"

def test_intentions_coverage():
    """Ensures test suite covers various workload scenarios."""
    scenarios = [
        {"project": "Diffusion-Gen", "type": "eval_benchmark", "tier": "tier_3_green_opportunistic"},
        {"project": "Kedro-Pipeline", "type": "batch_inference", "tier": "tier_2_standard"},
    ]
    for sc in scenarios:
        item = WorkloadRequestAnalysis(
            project_name=sc["project"],
            workload_type=sc["type"],
            priority_tier=sc["tier"],
            sla_deadline="flexible",
            semantic_summary=f"Automated test for {sc['project']}"
        )
        assert item.project_name == sc["project"]
