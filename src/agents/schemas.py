from typing import TypedDict, Any, Dict, Optional
from pydantic import BaseModel, Field

class WorkloadRequestAnalysis(BaseModel):
    """
    Structured extraction schema for AI compute requests and team scheduling requirements.
    """
    project_name: str = Field(
        description="Name of the project or model being scheduled (e.g. 'CrossViT-Botanical', 'Llama-Finotune', 'Data-Pipeline')"
    )
    workload_type: str = Field(
        description="Category of workload (e.g. 'llm_training', 'vision_transformers', 'batch_inference', 'eval_benchmark')"
    )
    gpus_requested: int = Field(
        default=4,
        description="Number of GPUs requested (clamped between 1 and 32)"
    )
    priority_tier: str = Field(
        description="SLA priority tier (e.g. 'tier_1_critical', 'tier_2_standard', 'tier_3_green_opportunistic')"
    )
    sla_deadline: str = Field(
        description="Required completion horizon or deadline (e.g. 'immediate', 'under_4h', 'overnight', 'flexible')"
    )
    semantic_summary: str = Field(
        description="Concise semantic summary of the compute requirement used for vector embedding and retrieval"
    )

class ClusterTelemetryState(TypedDict):
    """
    State contract for real-time cluster telemetry monitoring and scheduling graph.
    """
    raw_telemetry: Dict[str, Any]
    team_context: Optional[str]
    scheduling_decision: Optional[str]
    final_response: str

class WorkloadGraphState(TypedDict):
    """
    State contract for AI engineer interaction, SLA parsing and vector persistence graph.
    """
    raw_user_input: str
    team_id: str
    analysis: Optional[Dict[str, Any]]
    mcp_add_response: Optional[str]
    mcp_context_response: Optional[str]
    final_answer: Optional[str]
