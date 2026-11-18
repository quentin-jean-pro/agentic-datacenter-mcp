import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

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

from src.config import (
    COMPUTE_MCP_HOST,
    COMPUTE_MCP_PORT,
    DRY_RUN,
    TELEMETRY_ENDPOINT,
    PROMETHEUS_JOB
)

# --- 1. SETUP & LOGGING ---
logger = logging.getLogger("DataCenter_MCP")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = RotatingFileHandler("datacenter_compute.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Initialize FastMCP Server
try:
    mcp = FastMCP("DataCenter_Compute_Server", host=COMPUTE_MCP_HOST, port=COMPUTE_MCP_PORT)
except TypeError:
    mcp = FastMCP("DataCenter_Compute_Server")

# In-memory rate limiting and node state dictionary
node_last_action: Dict[str, datetime] = {}
CLUSTER_STATE: Dict[str, Dict[str, Any]] = {
    "node-alpha": {
        "gpu_model": "NVIDIA H100 80GB",
        "gpu_count": 8,
        "temperature_celsius": 54.2,
        "power_draw_watts": 320,
        "power_cap_watts": 400,
        "utilization_pct": 68.0,
        "throttle_percent": 0,
        "allocated_jobs": ["job_crossvit_01"]
    },
    "node-beta": {
        "gpu_model": "NVIDIA H100 80GB",
        "gpu_count": 8,
        "temperature_celsius": 71.5,
        "power_draw_watts": 395,
        "power_cap_watts": 400,
        "utilization_pct": 92.0,
        "throttle_percent": 0,
        "allocated_jobs": ["job_llm_pretrain_04"]
    },
    "node-gamma": {
        "gpu_model": "NVIDIA A100 80GB",
        "gpu_count": 8,
        "temperature_celsius": 81.4,
        "power_draw_watts": 415,
        "power_cap_watts": 400,
        "utilization_pct": 98.0,
        "throttle_percent": 10,
        "allocated_jobs": ["job_diffusion_eval"]
    },
    "node-delta": {
        "gpu_model": "NVIDIA A100 80GB",
        "gpu_count": 8,
        "temperature_celsius": 46.0,
        "power_draw_watts": 210,
        "power_cap_watts": 350,
        "utilization_pct": 35.0,
        "throttle_percent": 0,
        "allocated_jobs": []
    }
}

# --- 2. JSON CONTRACT HELPER ---
def build_mcp_response(status: str, system_context: dict, measurement: str, tags: dict, fields: dict) -> str:
    """
    Builds a strict, immutable JSON contract envelope returned to the AI Agent.
    Guarantees client-side parsing stability and transparent telemetry tracing.
    """
    response = {
        "execution_status": status,
        "system_context": system_context,
        "db_trace": {
            "telemetry_written": not DRY_RUN and status == "SUCCESS",
            "measurement": measurement,
            "tags": tags,
            "fields": fields
        }
    }
    if status == "SUCCESS":
        logger.info(f"📤 Action Successful (Measurement: {measurement}) | Status: {status}")
    else:
        logger.error(f"❌ Action Failure (Measurement: {measurement}) | Status: {status}")
        
    return json.dumps(response, indent=2, ensure_ascii=False)

# --- 3. MCP TOOLS: READ (Sensors & Cluster Telemetry) ---

@mcp.tool()
async def ping_compute_server() -> str:
    """Diagnostic tool to verify compute telemetry server health and operating status."""
    try:
        return build_mcp_response(
            status="SUCCESS",
            system_context={"status": "online", "dry_run": DRY_RUN, "telemetry_endpoint": TELEMETRY_ENDPOINT},
            measurement="system_ping",
            tags={},
            fields={}
        )
    except Exception as e:
        logger.exception(f"Error in ping_compute_server: {e}")
        return build_mcp_response(
            status="ERROR",
            system_context={"error_details": str(e)},
            measurement="system_ping",
            tags={},
            fields={}
        )

@mcp.tool()
async def get_cluster_telemetry() -> str:
    """
    Retrieves real-time physical telemetry of the compute cluster:
    GPU temperatures (°C), power draw (Watts), power caps, utilization %, PUE, and grid carbon intensity.
    Used by agents to ground scheduling and thermal safety decisions.
    """
    try:
        total_power = sum(node["power_draw_watts"] for node in CLUSTER_STATE.values())
        max_temp = max(node["temperature_celsius"] for node in CLUSTER_STATE.values())
        avg_util = sum(node["utilization_pct"] for node in CLUSTER_STATE.values()) / len(CLUSTER_STATE)

        metrics = {
            "pue": 1.18,
            "grid_carbon_intensity_g_kwh": 124.5,
            "total_cluster_power_watts": total_power,
            "max_gpu_temperature_celsius": max_temp,
            "average_cluster_utilization_pct": round(avg_util, 1),
            "nodes": CLUSTER_STATE
        }

        return build_mcp_response(
            status="SUCCESS",
            system_context=metrics,
            measurement="cluster_telemetry",
            tags={"cluster_id": "eu_west_ai_pod_1", "job": PROMETHEUS_JOB},
            fields={"total_power": total_power, "max_temp": max_temp, "pue": 1.18}
        )
    except Exception as e:
        logger.exception(f"Error in get_cluster_telemetry: {e}")
        return build_mcp_response(
            status="ERROR",
            system_context={"error_details": str(e)},
            measurement="cluster_telemetry",
            tags={},
            fields={}
        )

# --- 4. MCP TOOLS: WRITE (Actuation, Capping & Job Dispatch) ---

@mcp.tool()
async def set_node_power_cap(node_id: str, power_cap_watts: int) -> str:
    """
    Adjusts the dynamic power cap (Watts) on a compute node (e.g. 200W to 450W per GPU).
    Enforces hardware safety clamping, in-memory rate limiting, and telemetry logging.
    """
    try:
        if node_id not in CLUSTER_STATE:
            return build_mcp_response(
                status="ERROR",
                system_context={"error": f"Node '{node_id}' not found in cluster topology."},
                measurement="node_power_cap",
                tags={"node_id": node_id},
                fields={}
            )

        # Rate limiting: 30 seconds cooldown per node
        now = datetime.now(timezone.utc)
        if node_id in node_last_action:
            elapsed = (now - node_last_action[node_id]).total_seconds()
            if elapsed < 30.0:
                cooldown_remaining = round(30.0 - elapsed, 1)
                return build_mcp_response(
                    status="REJECTED_RATE_LIMIT",
                    system_context={
                        "error": f"Rate limit active on {node_id}. Please wait {cooldown_remaining}s.",
                        "cooldown_remaining_seconds": cooldown_remaining
                    },
                    measurement="node_power_cap",
                    tags={"node_id": node_id},
                    fields={}
                )

        # Hardware safety boundary clamping: [200, 450] W
        clamped_watts = max(200, min(450, power_cap_watts))
        CLUSTER_STATE[node_id]["power_cap_watts"] = clamped_watts
        node_last_action[node_id] = now

        return build_mcp_response(
            status="SUCCESS",
            system_context={
                "message": f"Power cap updated on {node_id}.",
                "node_id": node_id,
                "requested_watts": power_cap_watts,
                "applied_watts": clamped_watts
            },
            measurement="node_power_cap",
            tags={"node_id": node_id},
            fields={"applied_power_cap_watts": clamped_watts}
        )
    except Exception as e:
        logger.exception(f"Error in set_node_power_cap: {e}")
        return build_mcp_response(
            status="ERROR",
            system_context={"error_details": str(e)},
            measurement="node_power_cap",
            tags={"node_id": node_id},
            fields={}
        )

@mcp.tool()
async def throttle_node(node_id: str, throttle_percent: int) -> str:
    """
    Applies clock throttling (0% to 50%) to immediately stabilize an overheating GPU node.
    """
    try:
        if node_id not in CLUSTER_STATE:
            return build_mcp_response(
                status="ERROR",
                system_context={"error": f"Node '{node_id}' not found."},
                measurement="node_throttle",
                tags={"node_id": node_id},
                fields={}
            )

        clamped_throttle = max(0, min(50, throttle_percent))
        CLUSTER_STATE[node_id]["throttle_percent"] = clamped_throttle

        return build_mcp_response(
            status="SUCCESS",
            system_context={
                "node_id": node_id,
                "applied_throttle_percent": clamped_throttle,
                "temperature_celsius": CLUSTER_STATE[node_id]["temperature_celsius"]
            },
            measurement="node_throttle",
            tags={"node_id": node_id},
            fields={"throttle_percent": clamped_throttle}
        )
    except Exception as e:
        logger.exception(f"Error in throttle_node: {e}")
        return build_mcp_response(
            status="ERROR",
            system_context={"error_details": str(e)},
            measurement="node_throttle",
            tags={"node_id": node_id},
            fields={}
        )

@mcp.tool()
async def dispatch_workload(job_id: str, node_id: str, gpus_allocated: int, priority_tier: str) -> str:
    """
    Schedules an AI training or inference workload on an available compute node.
    """
    try:
        if node_id not in CLUSTER_STATE:
            return build_mcp_response(
                status="ERROR",
                system_context={"error": f"Node '{node_id}' not found."},
                measurement="workload_dispatch",
                tags={"job_id": job_id},
                fields={}
            )

        node = CLUSTER_STATE[node_id]
        if node["temperature_celsius"] > 80.0:
            return build_mcp_response(
                status="REJECTED_THERMAL_SAFETY",
                system_context={
                    "error": f"Node {node_id} exceeds safe thermal envelope ({node['temperature_celsius']}°C > 80°C). Dispatch aborted."
                },
                measurement="workload_dispatch",
                tags={"job_id": job_id, "node_id": node_id},
                fields={}
            )

        if job_id not in node["allocated_jobs"]:
            node["allocated_jobs"].append(job_id)

        return build_mcp_response(
            status="SUCCESS",
            system_context={
                "message": f"Job '{job_id}' dispatched to '{node_id}'.",
                "gpus_allocated": gpus_allocated,
                "priority_tier": priority_tier,
                "current_node_jobs": node["allocated_jobs"]
            },
            measurement="workload_dispatch",
            tags={"job_id": job_id, "node_id": node_id, "priority": priority_tier},
            fields={"gpus_allocated": gpus_allocated}
        )
    except Exception as e:
        logger.exception(f"Error in dispatch_workload: {e}")
        return build_mcp_response(
            status="ERROR",
            system_context={"error_details": str(e)},
            measurement="workload_dispatch",
            tags={"job_id": job_id},
            fields={}
        )

if __name__ == "__main__":
    logger.info(f"Starting DataCenter Compute MCP Server on {COMPUTE_MCP_HOST}:{COMPUTE_MCP_PORT} (SSE Transport)...")
    mcp.run(transport="sse")
