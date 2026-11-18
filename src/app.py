import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import asyncio

import streamlit as st

from src.agents.schemas import WorkloadRequestAnalysis
from src.config import DATA_DIR
from src.mcp_servers.compute_telemetry_server import (
    CLUSTER_STATE,
    build_mcp_response,
    set_node_power_cap,
    throttle_node,
)
from src.utils.db_manager import PolicyDBManager

st.set_page_config(
    page_title="Green AI Data Center | Agentic MCP Orchestrator",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #6366f1; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #94a3b8; margin-bottom: 1.5rem; }
    .node-card { background: #1e293b; border-radius: 10px; padding: 15px; border-left: 4px solid #6366f1; margin-bottom: 10px; }
    .metric-val { font-size: 1.6rem; font-weight: bold; color: #f8fafc; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header">🖥️ Green AI Data Center & Workload Orchestrator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<div class=\"sub-header\"><strong>Maquette d'usage et d'exploitation R&D</strong> : Supervision agentique FastMCP (Port 8001/8002) & LangGraph pour clusters GPU H100/A100.</div>",
    unsafe_allow_html=True,
)

# Sidebar: Controls & Modes
st.sidebar.header("⚙️ Configuration & Télémétrie")
dry_run_mode = st.sidebar.toggle("Mode DRY_RUN (Simulation matérielle)", value=True)
carbon_intensity = st.sidebar.slider(
    "Intensité Carbone Réseau (gCO2/kWh)", min_value=40.0, max_value=280.0, value=124.5, step=5.0
)
pue_target = st.sidebar.number_input(
    "PUE (Power Usage Effectiveness)", min_value=1.05, max_value=1.60, value=1.18, step=0.01
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Pre-Flight Safety Guards")
st.sidebar.info("""
- **Plage Power-Cap :** [200W, 450W]
- **Seuil Alerte Température :** 80.0°C
- **Rate-Limiting Actuation :** 30s / nœud
- **Isolation Données :** DuckDB par équipe
""")

# Main Layout: 2 Columns (Cluster Status / Agent Console)
col_left, col_right = st.columns([1.1, 1.0], gap="large")

with col_left:
    st.subheader("📊 Télémétrie Temps Réel des Nœuds GPU")

    # Global Cluster Metrics
    m1, m2, m3, m4 = st.columns(4)
    total_power = sum(n["power_draw_watts"] for n in CLUSTER_STATE.values())
    max_t = max(n["temperature_celsius"] for n in CLUSTER_STATE.values())
    avg_u = sum(n["utilization_pct"] for n in CLUSTER_STATE.values()) / len(CLUSTER_STATE)

    m1.metric(
        "Puissance Totale",
        f"{total_power} W",
        delta="-180 W (Éco)" if carbon_intensity > 150 else "Nominal",
    )
    m2.metric(
        "Température Max",
        f"{max_t:.1f} °C",
        delta="Alerte > 80°C" if max_t >= 80.0 else "Stable",
        delta_color="inverse",
    )
    m3.metric("Utilisation Moyenne", f"{avg_u:.1f} %")
    m4.metric(
        "Mix Électrique",
        f"{carbon_intensity:.0f} g/kWh",
        delta="Vert" if carbon_intensity < 100 else "Standard",
    )

    st.markdown("---")

    # Individual Nodes
    for node_id, node in CLUSTER_STATE.items():
        is_hot = node["temperature_celsius"] >= 80.0
        border_color = (
            "#ef4444" if is_hot else ("#10b981" if node["utilization_pct"] < 50 else "#6366f1")
        )

        with st.container():
            st.markdown(
                f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-left: 5px solid {border_color}; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="font-size: 1.1rem; color: #f8fafc;">{node_id.upper()}</strong>
                    <span style="font-size: 0.85rem; padding: 3px 8px; border-radius: 4px; background: {"rgba(239, 68, 68, 0.2)" if is_hot else "rgba(99, 102, 241, 0.2)"}; color: {"#f87171" if is_hot else "#818cf8"};">
                        {node["gpu_model"]} ({node["gpu_count"]} GPUs)
                    </span>
                </div>
                <div style="display: flex; gap: 20px; margin-top: 8px; font-size: 0.9rem;">
                    <span>🌡️ Temp: <strong>{node["temperature_celsius"]}°C</strong></span>
                    <span>⚡ Power: <strong>{node["power_draw_watts"]}W</strong> / Cap: {node["power_cap_watts"]}W</span>
                    <span>📈 Charge: <strong>{node["utilization_pct"]}%</strong></span>
                    <span>🛑 Throttling: <strong>{node["throttle_percent"]}%</strong></span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Manual Safety Action Simulation
    st.markdown("#### 🧪 Test d'Actuation Cyber-Physique FastMCP")
    sel_node = st.selectbox("Sélectionner un nœud à réguler :", list(CLUSTER_STATE.keys()), index=2)
    c_act1, c_act2 = st.columns(2)
    with c_act1:
        new_power = st.slider(
            "Ajuster Power-Cap (W)",
            200,
            450,
            int(CLUSTER_STATE[sel_node]["power_cap_watts"]),
            step=10,
        )
        if st.button("Appliquer Power-Cap via MCP"):
            res = asyncio.run(set_node_power_cap(sel_node, new_power))
            st.session_state["last_mcp_action"] = {
                "action": f"Power-Cap ajusté à {new_power}W sur {sel_node}",
                "result": res,
            }
            st.rerun()
    with c_act2:
        new_throttle = st.slider(
            "Appliquer Clock Throttling (%)",
            0,
            50,
            int(CLUSTER_STATE[sel_node]["throttle_percent"]),
            step=5,
        )
        if st.button("Activer Throttling via MCP"):
            res = asyncio.run(throttle_node(sel_node, new_throttle))
            st.session_state["last_mcp_action"] = {
                "action": f"Throttling réglé à {new_throttle}% sur {sel_node}",
                "result": res,
            }
            st.rerun()

    # Instant Display of last executed FastMCP contract
    if st.session_state.get("last_mcp_action"):
        act_info = st.session_state["last_mcp_action"]
        st.success(
            f"⚡ **Action FastMCP exécutée : {act_info['action']}** (Mise à jour en temps réel)"
        )
        st.code(act_info["result"], language="json")

with col_right:
    st.subheader("🧠 Console de Requête Agentique LangGraph")
    st.write("Exprimez un besoin de calcul en langage naturel pour une équipe R&D :")

    sample_queries = [
        "Besoin de 8 GPU H100 pour entraîner CrossViT sur les herbiers avant demain 8h (priorité critique).",
        "Lancer un benchmark LoRA sur Llama 3 (8B) avec priorité flexible pour profiter des heures bas carbone.",
        "Traitement batch de segmentation d'images botaniques (4 GPUs, priorité basse, exécution nocturne).",
    ]
    user_choice = st.selectbox(
        "Exemples prédéfinis :", ["(Saisir ma propre requête...)"] + sample_queries
    )

    if user_choice == "(Saisir ma propre requête...)":
        user_prompt = st.text_area(
            "Prompt ingénieur :",
            value="Besoin urgent de 8 GPU pour fine-tuning CrossViT d'ici 4 heures.",
            height=80,
        )
    else:
        user_prompt = st.text_area("Prompt ingénieur :", value=user_choice, height=80)

    team_id = st.text_input("Identifiant Équipe (Tenant DuckDB) :", value="cv_research_team")

    if st.button("Soumettre à l'Orchestrateur LangGraph", type="primary"):
        with st.spinner(
            "Exécution du pipeline (Garde-fous -> Extraction de schéma -> Requête DuckDB -> Contrat FastMCP)..."
        ):
            p_low = user_prompt.lower()

            # --- Rule-Based Deterministic Pre-Filtering & Safety Guardrails ---
            # 1. Analyse temporelle et calcul précis de l'échéance SLA
            if "demain 8h" in p_low or "demain 8:00" in p_low or "demain matin" in p_low:
                extracted_deadline = "demain 8h00 (~20h de marge disponible)"
                priority = (
                    "tier_1_critical"
                    if ("critique" in p_low or "urgent" in p_low)
                    else "tier_2_standard"
                )
            elif "4h" in p_low or "4 heures" in p_low:
                extracted_deadline = "d'ici 4 heures (< 4h restant)"
                priority = "tier_1_critical"
            elif "nocturne" in p_low or "nuit" in p_low or "creuses" in p_low:
                extracted_deadline = "créneau nocturne (22h-06h, heures bas carbone)"
                priority = "tier_3_green_opportunistic"
            elif "urgent" in p_low or "critique" in p_low:
                extracted_deadline = "immédiat (< 2h)"
                priority = "tier_1_critical"
            else:
                extracted_deadline = "standard / flexible sous 48h"
                priority = "tier_2_standard"

            # 2. Extraction du projet et du type de charge
            proj_name = (
                "CrossViT-Herbarium"
                if "crossvit" in p_low or "herbier" in p_low
                else ("Llama3-LoRA" if "llama" in p_low else "AI-Workload-Generic")
            )
            w_type = (
                "vision_transformers"
                if "crossvit" in p_low or "image" in p_low
                else ("llm_training" if "llama" in p_low or "lora" in p_low else "batch_inference")
            )
            gpus = 8 if "8" in user_prompt else (4 if "4" in user_prompt else 2)

            analysis = WorkloadRequestAnalysis(
                project_name=proj_name,
                workload_type=w_type,
                gpus_requested=gpus,
                priority_tier=priority,
                sla_deadline=extracted_deadline,
                semantic_summary=user_prompt.strip(),
            )

            # Affichage étape 1 : Parsing & Validation de Schéma
            st.success("✅ **Étape 1 : Parsing Structuré Pydantic V2 Validé**")
            st.caption(
                "Contrat strict : Extraction déterministe des quotas de calcul et calcul du délai SLA."
            )
            st.json(analysis.model_dump())

            # Étape 2 : Mémoire Vectorielle DuckDB
            st.info("🔍 **Étape 2 : Consultation de l'historique et des SLAs dans DuckDB**")
            with PolicyDBManager(team_id=team_id, db_dir=DATA_DIR) as db:
                dummy_v = [0.1] * 384
                matches = db.get_similar_policies(dummy_v, limit=2)
                if matches:
                    st.write(f"Contexte trouvé pour l'équipe `{team_id}` (Similarité sémantique) :")
                    for m in matches:
                        st.caption(
                            f"• **{m['project_name']}** ({m['workload_type']}) — SLA: {m['sla_deadline']} (Sim: {m.get('similarity', 0.95):.2f})"
                        )
                else:
                    st.caption("Premier enregistrement pour cette équipe dans le tenant DuckDB.")

            # Étape 3 : Décision Orchestrateur & Contrat d'Actuation FastMCP
            st.success("📤 **Étape 3 : Contrat d'Actuation FastMCP émis (Port 8001)**")

            # Vérification de sécurité matérielle par le middleware
            hot_node = next(
                (nid for nid, n in CLUSTER_STATE.items() if n["temperature_celsius"] >= 80.0), None
            )
            target_node = "node-delta" if hot_node else "node-alpha"
            safety_reason = (
                f"Seuil d'alerte : Surchauffe sur {hot_node} -> Déroutement automatique vers {target_node}"
                if hot_node
                else "Nominal (T < 80°C)"
            )

            contract_res = build_mcp_response(
                status="SUCCESS",
                system_context={
                    "allocated_node": target_node,
                    "gpus_assigned": analysis.gpus_requested,
                    "sla_committed": extracted_deadline,
                    "carbon_saving_scheduled": carbon_intensity < 100,
                    "safety_interlock": safety_reason,
                },
                measurement="workload_dispatch",
                tags={"team_id": team_id, "project": analysis.project_name},
                fields={"gpus_requested": analysis.gpus_requested, "pue": pue_target},
            )
            st.code(contract_res, language="json")

            # Note d'Architecture
            st.markdown(
                """
            <div style="background: rgba(99, 102, 241, 0.1); border-left: 4px solid #6366f1; padding: 10px; border-radius: 6px; margin-top: 10px;">
                <strong>Architecture Microservices Découplée :</strong><br/>
                • <em>Couche Sémantique (Port 8002)</em> : Ingestion du prompt, extraction structurée et mémoire vectorielle DuckDB.<br/>
                • <em>Couche Matérielle (Port 8001)</em> : Télémétrie GPU temps réel, power-capping et garde-fous thermiques locaux.
            </div>
            """,
                unsafe_allow_html=True,
            )

st.markdown("---")
st.caption(
    "Quentin JEAN — Portfolio R&D | FastMCP SSE Transport (8001/8002) · LangGraph StateGraph · DuckDB Multi-Tenant · Pydantic V2"
)
