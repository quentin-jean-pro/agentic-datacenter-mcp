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

from src.config import COMPUTE_MCP_URL, GOOGLE_API_KEY, GROQ_API_KEY
from src.agents.schemas import ClusterTelemetryState

# --- LLM Client Initialization ---
def get_llm():
    if GOOGLE_API_KEY and GOOGLE_API_KEY != "your_gemini_api_key_here":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
    elif GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY)
    else:
        # Standalone deterministic mock LLM for offline testing
        class MockLLM:
            async def ainvoke(self, prompt: str):
                class Response:
                    content = (
                        "Cluster telemetry indicates node-delta has 8 available GPUs at 46.0°C. "
                        "Node-gamma is running hot at 81.4°C with 10% active clock throttling. "
                        "Recommendation: Schedule new training workloads on node-delta to minimize thermal wear "
                        "and capitalize on green off-peak grid carbon intensity."
                    )
                return Response()
        return MockLLM()

llm = get_llm()

# --- LangGraph Nodes ---

async def telemetry_polling_node(state: ClusterTelemetryState) -> dict:
    """
    Connects to the FastMCP Compute Server over SSE, queries cluster metrics,
    and extracts structured telemetry.
    """
    print(f"\n[Node: telemetry_polling] Connecting to FastMCP Compute Server ({COMPUTE_MCP_URL})...")
    try:
        async with sse_client(COMPUTE_MCP_URL) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print("[Node: telemetry_polling] MCP Handshake successful. Invoking 'get_cluster_telemetry'...")
                mcp_result = await session.call_tool(name="get_cluster_telemetry", arguments={})
                raw_text = mcp_result.content[0].text if mcp_result.content else "{}"
                telemetry_data = json.loads(raw_text)
                return {"raw_telemetry": telemetry_data}
    except Exception as e:
        print(f"[Node: telemetry_polling] Warning: Could not connect to live MCP server ({e}). Using offline telemetry mock.")
        mock_data = {
            "execution_status": "SUCCESS",
            "system_context": {
                "pue": 1.18,
                "grid_carbon_intensity_g_kwh": 124.5,
                "total_cluster_power_watts": 1340,
                "max_gpu_temperature_celsius": 81.4,
                "average_cluster_utilization_pct": 73.2,
                "nodes": {
                    "node-alpha": {"temperature_celsius": 54.2, "utilization_pct": 68.0, "power_draw_watts": 320},
                    "node-beta": {"temperature_celsius": 71.5, "utilization_pct": 92.0, "power_draw_watts": 395},
                    "node-gamma": {"temperature_celsius": 81.4, "utilization_pct": 98.0, "power_draw_watts": 415},
                    "node-delta": {"temperature_celsius": 46.0, "utilization_pct": 35.0, "power_draw_watts": 210}
                }
            }
        }
        return {"raw_telemetry": mock_data}

async def scheduling_reasoning_node(state: ClusterTelemetryState) -> dict:
    """
    Analyzes cluster metrics with LLM reasoning to determine optimal job allocation and thermal safety actions.
    """
    print("[Node: scheduling_reasoning] Synthesizing telemetry and safety envelopes...")
    context_str = json.dumps(state.get("raw_telemetry", {}), indent=2)
    prompt = (
        "You are an AI Cluster Orchestrator. Review the following live GPU telemetry:\n"
        f"{context_str}\n\n"
        "Assess thermal limits, compute headroom, and generate an optimal scheduling recommendation."
    )
    response = await llm.ainvoke(prompt)
    return {
        "scheduling_decision": response.content,
        "final_response": response.content
    }

# --- StateGraph Assembly ---

workflow = StateGraph(ClusterTelemetryState)
workflow.add_node("telemetry_polling", telemetry_polling_node)
workflow.add_node("scheduling_reasoning", scheduling_reasoning_node)

workflow.set_entry_point("telemetry_polling")
workflow.add_edge("telemetry_polling", "scheduling_reasoning")
workflow.add_edge("scheduling_reasoning", END)

orchestrator_app = workflow.compile()

async def run_orchestrator():
    initial_state: ClusterTelemetryState = {
        "raw_telemetry": {},
        "team_context": None,
        "scheduling_decision": None,
        "final_response": ""
    }
    print("🚀 Starting DataCenter Orchestrator LangGraph Execution...")
    result = await orchestrator_app.ainvoke(initial_state)
    print("\n✅ Final Cluster Decision:")
    print(result.get("final_response"))
    return result

if __name__ == "__main__":
    asyncio.run(run_orchestrator())
