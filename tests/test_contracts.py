import json

from src.mcp_servers.compute_telemetry_server import build_mcp_response


def test_build_mcp_response_success():
    context = {"node_id": "node-alpha", "power_cap_watts": 350}
    tags = {"cluster": "eu_west"}
    fields = {"applied_power_cap_watts": 350}
    raw = build_mcp_response("SUCCESS", context, "node_power_cap", tags, fields)
    parsed = json.loads(raw)

    assert parsed["execution_status"] == "SUCCESS"
    assert parsed["system_context"]["node_id"] == "node-alpha"
    assert parsed["db_trace"]["measurement"] == "node_power_cap"
    assert parsed["db_trace"]["tags"] == tags
    assert parsed["db_trace"]["fields"] == fields


def test_build_mcp_response_error():
    raw = build_mcp_response("ERROR", {"error": "thermal_spike"}, "node_throttle", {}, {})
    parsed = json.loads(raw)

    assert parsed["execution_status"] == "ERROR"
    assert parsed["db_trace"]["telemetry_written"] is False
