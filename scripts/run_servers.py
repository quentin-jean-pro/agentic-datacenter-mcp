"""
Convenience launcher to run both FastMCP Data Center servers concurrently.
"""

import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    compute_path = root / "src" / "mcp_servers" / "compute_telemetry_server.py"
    policy_path = root / "src" / "mcp_servers" / "workload_policy_server.py"

    print("🚀 Launching FastMCP Compute Telemetry Server (Port 8001)...")
    p1 = subprocess.Popen([sys.executable, str(compute_path)])

    print("🚀 Launching FastMCP Workload Policy Vector Server (Port 8002)...")
    p2 = subprocess.Popen([sys.executable, str(policy_path)])

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Terminating FastMCP servers cleanly...")
        p1.terminate()
        p2.terminate()
        p1.wait()
        p2.wait()
        print("Done.")


if __name__ == "__main__":
    main()
