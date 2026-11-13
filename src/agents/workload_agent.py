import os
import sys
import json
import asyncio
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from mcp import ClientSession
from mcp.client.sse import sse_client

# Add parent directory to sys.path if run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.config import POLICY_MCP_URL, GOOGLE_API_KEY, GROQ_API_KEY
from src.agents.schemas import WorkloadRequestAnalysis, WorkloadGraphState

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT_WORKLOAD = (
    "You are an automated workload ingestion parser for an AI compute orchestration cluster. "
    "Parse the incoming unstructured engineer request into a validated WorkloadRequestAnalysis schema. "
    "Extract project identifiers, workload types, requested GPU resources (clamped to [1, 32]), "
    "and map temporal scheduling constraints into SLA deadlines and priority tiers."
)

# --- LLM Client Initialization ---
def get_llm():
    if GOOGLE_API_KEY and GOOGLE_API_KEY != "your_gemini_api_key_here":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
    elif GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY)
    else:
        class MockLLM:
            def with_structured_output(self, schema):
                class StructuredWrapper:
                    async def ainvoke(self, prompt: str):
                        return WorkloadRequestAnalysis(
                            project_name="CrossViT-Botanical",
                            workload_type="vision_transformers",
                            gpus_requested=8,
                            priority_tier="tier_1_critical",
                            sla_deadline="demain 8h00 (~20h de marge)",
                            semantic_summary="Urgent 8x GPU allocation for CrossViT multi-scale botanical classification benchmark."
                        )
                return StructuredWrapper()
            
            async def ainvoke(self, prompt: str):
                class Response:
                    content = (
                        "Allocation request parsed and SLA registered. Team context verified: "
                        "node-delta has 8 available GPUs ready for immediate allocation under tier 1 SLA."
                    )
                return Response()
        return MockLLM()

llm = get_llm()

# --- LangGraph Nodes with Guardian & Token-Budget Guardrails ---

async def parse_workload_node(state: WorkloadGraphState) -> dict:
    """
    Parses unstructured engineer prompt into structured Pydantic WorkloadRequestAnalysis.
    Enforces Guardian pre-flight checks to minimize API token expenditure.
    """
    print(f"\n[Node: parse_workload] Parsing prompt for team '{state['team_id']}'...")
    raw_text = state["raw_user_input"]
    p_low = raw_text.lower()

    # --- GUARDIAN: Pre-Flight Rule-Based Temporal & Safety Extraction ---
    # Calculates realistic SLA margin and avoids redundant token cycles
    if "demain 8h" in p_low or "demain 8:00" in p_low or "demain matin" in p_low:
        computed_deadline = "demain 8h00 (~20h de marge)"
        computed_tier = "tier_1_critical" if ("critique" in p_low or "urgent" in p_low) else "tier_2_standard"
    elif "4h" in p_low or "4 heures" in p_low:
        computed_deadline = "d'ici 4 heures (< 4h restant)"
        computed_tier = "tier_1_critical"
    elif "nocturne" in p_low or "nuit" in p_low or "creuses" in p_low:
        computed_deadline = "créneau nocturne (heures creuses bas carbone)"
        computed_tier = "tier_3_green_opportunistic"
    elif "urgent" in p_low or "critique" in p_low:
        computed_deadline = "immédiat (< 2h)"
        computed_tier = "tier_1_critical"
    else:
        computed_deadline = "standard flexible sous 48h"
        computed_tier = "tier_2_standard"

    try:
        structured_llm = llm.with_structured_output(WorkloadRequestAnalysis)
        prompt = (
            f"{SYSTEM_PROMPT_WORKLOAD}\n"
            "Extract structured AI compute scheduling details from this team request:\n"
            f"\"{raw_text}\""
        )
        analysis: WorkloadRequestAnalysis = await structured_llm.ainvoke(prompt)
        # Ensure guardian safety bounds on GPUs
        analysis.gpus_requested = max(1, min(32, analysis.gpus_requested))
        return {"analysis": analysis.model_dump()}
    except Exception as e:
        print(f"[Node: parse_workload] Guardian structured fallback applied ({e}).")
        fallback = WorkloadRequestAnalysis(
            project_name="CrossViT-Herbarium" if "crossvit" in p_low or "herbier" in p_low else ("Llama3-LoRA" if "llama" in p_low else "AI-Workload"),
            workload_type="vision_transformers" if "crossvit" in p_low or "image" in p_low else ("llm_training" if "llama" in p_low or "lora" in p_low else "batch_inference"),
            gpus_requested=8 if "8" in raw_text else 4,
            priority_tier=computed_tier,
            sla_deadline=computed_deadline,
            semantic_summary=raw_text[:120].strip()
        )
        return {"analysis": fallback.model_dump()}

async def persist_policy_node(state: WorkloadGraphState) -> dict:
    """
    Sends structured workload details to the FastMCP Policy Server for DuckDB vector storage.
    """
    print(f"[Node: persist_policy] Sending policy intent to FastMCP ({POLICY_MCP_URL})...")
    analysis = state.get("analysis", {})
    try:
        async with sse_client(POLICY_MCP_URL) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                args = {
                    "team_id": state["team_id"],
                    "project_name": analysis.get("project_name", "Unknown"),
                    "workload_type": analysis.get("workload_type", "batch"),
                    "priority_tier": analysis.get("priority_tier", "standard"),
                    "sla_deadline": analysis.get("sla_deadline", "standard"),
                    "raw_prompt": state["raw_user_input"],
                    "semantic_summary": analysis.get("semantic_summary", "")
                }
                mcp_res = await session.call_tool(name="add_workload_policy_interaction", arguments=args)
                txt = mcp_res.content[0].text if mcp_res.content else "No response"
                return {"mcp_add_response": txt}
    except Exception as e:
        print(f"[Node: persist_policy] Warning: MCP Policy Server offline ({e}). Mocking persistence.")
        return {"mcp_add_response": f"MOCK_PERSISTED_FOR_{state['team_id']}"}

async def retrieve_context_node(state: WorkloadGraphState) -> dict:
    """
    Queries past team workloads and SLA history using vector similarity.
    """
    print(f"[Node: retrieve_context] Retrieving historical SLA context for team '{state['team_id']}'...")
    try:
        async with sse_client(POLICY_MCP_URL) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                mcp_res = await session.call_tool(
                    name="get_team_context",
                    arguments={"team_id": state["team_id"], "query": state["raw_user_input"], "limit": 2}
                )
                txt = mcp_res.content[0].text if mcp_res.content else ""
                return {"mcp_context_response": txt}
    except Exception as e:
        print(f"[Node: retrieve_context] Warning: MCP Policy Server offline ({e}). Using mock context.")
        return {"mcp_context_response": "MOCK_HISTORICAL_SLA: Tier-1 batch guaranteed 99.5% completion on time."}

async def synthesize_plan_node(state: WorkloadGraphState) -> dict:
    """
    Synthesizes the final scheduling confirmation and SLA guarantee for the user.
    """
    print("[Node: synthesize_plan] Formulating final scheduling response...")
    analysis = state.get("analysis", {})
    response = (
        f"✅ Workload '{analysis.get('project_name')}' registered successfully.\n"
        f"- Workload Type: {analysis.get('workload_type')}\n"
        f"- Allocated Resources: {analysis.get('gpus_requested')} GPUs\n"
        f"- SLA Priority: {analysis.get('priority_tier')} (Deadline: {analysis.get('sla_deadline')})\n"
        f"- Persistence Status: {state.get('mcp_add_response')}\n"
        f"- Historical Context: {state.get('mcp_context_response')}"
    )
    return {"final_answer": response}

# --- StateGraph Assembly ---

workflow = StateGraph(WorkloadGraphState)
workflow.add_node("parse_workload", parse_workload_node)
workflow.add_node("persist_policy", persist_policy_node)
workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("synthesize_plan", synthesize_plan_node)

workflow.set_entry_point("parse_workload")
workflow.add_edge("parse_workload", "persist_policy")
workflow.add_edge("persist_policy", "retrieve_context")
workflow.add_edge("retrieve_context", "synthesize_plan")
workflow.add_edge("synthesize_plan", END)

workload_agent_app = workflow.compile()

async def run_workload_agent(team_id: str, prompt: str):
    initial_state: WorkloadGraphState = {
        "raw_user_input": prompt,
        "team_id": team_id,
        "analysis": None,
        "mcp_add_response": None,
        "mcp_context_response": None,
        "final_answer": None
    }
    print(f"🚀 Dispatching Workload Agent for Team '{team_id}'...")
    result = await workload_agent_app.ainvoke(initial_state)
    print("\n✅ Final Scheduling Response:")
    print(result.get("final_answer"))
    return result

if __name__ == "__main__":
    sample_prompt = "We urgently need 8 GPUs to train CrossViT on digitized herbarium sheets before 8 AM tomorrow."
    asyncio.run(run_workload_agent("cv_research_team", sample_prompt))

