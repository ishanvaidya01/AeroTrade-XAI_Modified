import streamlit as st
import os
import json
from dotenv import load_dotenv
from scenarios import SCENARIOS
from agent import run_agent, run_self_critique, answer_followup
from ui_components import render_agent_trace, render_tradeoff_matrix, render_rejection_panel, render_executive_summary, render_route_map, render_sensitivity_chart
from models import ToolTrace, RouteOption

load_dotenv()

st.set_page_config(page_title="AeroTrade XAI", layout="wide")
st.title("AeroTrade XAI: Autonomous Supply Chain Orchestrator")

st.sidebar.subheader("Route Selection")

origins = set()
destinations = set()
for s in SCENARIOS:
    for r in s["routes"]:
        origins.add(r.origin)
        destinations.add(r.destination)

selected_origin = st.sidebar.selectbox("Select Origin", sorted(list(origins)), index=sorted(list(origins)).index("Shenzhen") if "Shenzhen" in origins else 0)
selected_destination = st.sidebar.selectbox("Select Destination", sorted(list(destinations)), index=sorted(list(destinations)).index("Frankfurt") if "Frankfurt" in destinations else 0)

relevant_scenarios = []
for s in SCENARIOS:
    matching_routes = [r for r in s["routes"] if r.origin == selected_origin and r.destination == selected_destination]
    if matching_routes:
        s_copy = s.copy()
        s_copy["routes"] = matching_routes
        relevant_scenarios.append(s_copy)

if not relevant_scenarios:
    from news import fetch_real_disruption
    real_news = fetch_real_disruption(selected_origin, selected_destination)
    
    if real_news:
        news_scenario = {
            "id": "news_" + selected_origin[:3] + "_" + selected_destination[:3],
            "name": f"Live Alert: {real_news['name']}",
            "description": f"⚠️ **Real-Time Disruption Detected:** [{real_news['description']}]({real_news['url']}). The AI orchestrator must rapidly evaluate alternative paths for the {selected_origin} to {selected_destination} corridor.",
            "routes": [
                RouteOption(id="n_air", name="Expedited Air Freight", mode="air", origin=selected_origin, destination=selected_destination),
                RouteOption(id="n_sea", name="Rerouted Sea Freight", mode="sea", origin=selected_origin, destination=selected_destination),
                RouteOption(id="n_rail", name="Overland Rail Express", mode="rail", origin=selected_origin, destination=selected_destination),
                RouteOption(id="n_road", name="Long-Haul Cross-border Trucking", mode="road", origin=selected_origin, destination=selected_destination),
                RouteOption(id="n_multi", name="Premium Sea-Air Hybrid", mode="sea-air", origin=selected_origin, destination=selected_destination)
            ]
        }
        relevant_scenarios = [news_scenario]
    else:
        bau_scenario = {
            "id": "bau_" + selected_origin[:3] + "_" + selected_destination[:3],
            "name": "Business as Usual (Baseline Assessment)",
            "description": f"No active disruption detected for the **{selected_origin}** to **{selected_destination}** corridor. The agent will run a baseline assessment to compare standard routing options and calculate optimal baseline prices.",
            "routes": [
                RouteOption(id="bau_air", name="Direct Air Freight", mode="air", origin=selected_origin, destination=selected_destination),
                RouteOption(id="bau_sea", name="Direct Sea Freight", mode="sea", origin=selected_origin, destination=selected_destination),
                RouteOption(id="bau_rail", name="Overland Rail Freight", mode="rail", origin=selected_origin, destination=selected_destination)
            ]
        }
        relevant_scenarios = [bau_scenario]

scenario_names = [s["name"] for s in relevant_scenarios]
selected_scenario_name = st.sidebar.selectbox("Active Disruption:", scenario_names)
selected_scenario = next(s for s in relevant_scenarios if s["name"] == selected_scenario_name)

st.sidebar.markdown("---")

st.sidebar.subheader("Strategic Priorities")
st.sidebar.caption("Adjust the sliders below to communicate your company's current risk appetite and operational goals to the AI orchestrator.")

weight_cost = st.sidebar.slider("Cost Weight", 0.0, 1.0, 0.4, help="High weight prioritizes minimizing capital expenditure and freight costs.")
weight_time = st.sidebar.slider("Speed Weight", 0.0, 1.0, 0.3, help="High weight prioritizes the fastest possible transit time, ignoring premium pricing.")
weight_risk = st.sidebar.slider("Reliability (Risk) Weight", 0.0, 1.0, 0.2, help="High weight prioritizes stable, safe corridors and penalizes routes traversing conflict zones or extreme weather.")
weight_carbon = st.sidebar.slider("Sustainability Weight", 0.0, 1.0, 0.1, help="High weight prioritizes ESG goals and penalizes high carbon-footprint transport modes like air freight.")

weights = {
    "cost": weight_cost,
    "time": weight_time,
    "risk": weight_risk,
    "carbon": weight_carbon
}

if selected_scenario["id"].startswith("bau"):
    st.success(f"**{selected_scenario['name']}:** {selected_scenario['description']}")
else:
    st.error(f"**Disruption Event ({selected_scenario['name']}):** {selected_scenario['description']}")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Agent Execution Log")
    trace_container = st.container()
    
    if st.button("Run Agent", type="primary"):
        st.session_state.run_triggered = True
        st.session_state.traces = []
        st.session_state.result = None
        st.session_state.critique = None
        st.session_state.followups = []

if getattr(st.session_state, "run_triggered", False):
    routes_dict = {r.id: r for r in selected_scenario["routes"]}
    
    with col1:
        if not st.session_state.result:
            progress_bar = st.progress(0, text="Initializing agent...")
            
            def trace_callback(trace: ToolTrace):
                render_agent_trace(trace_container, trace)
                st.session_state.traces.append(trace)
                progress = min(len(st.session_state.traces) / (len(selected_scenario["routes"]) * 4), 1.0)
                progress_bar.progress(progress, text=f"Evaluating routes... ({len(st.session_state.traces)} tool calls)")
                
            with st.spinner("Agent is evaluating options..."):
                result = run_agent(selected_scenario, selected_scenario["routes"], trace_callback)
                st.session_state.result = result
            
            with st.spinner("Agent generating self-critique..."):
                st.session_state.critique = run_self_critique(result)
                
            progress_bar.empty()
            st.rerun()
            
        else:
            routes_to_traces = {}
            for t in st.session_state.traces:
                rid = t.input_args.get("route_id", "Unknown")
                routes_to_traces.setdefault(rid, []).append(t)
                
            for rid, r_traces in routes_to_traces.items():
                r_name = routes_dict.get(rid, selected_scenario["routes"][0]).name if rid in routes_dict else rid
                with st.expander(f"Route: {r_name} ▸ {len(r_traces)} tool calls"):
                    for t in r_traces:
                        st.write(f"**{t.tool_name}** | Output: `{t.output}`")
                        
            st.subheader("Optimized Route Map")
            chosen_route_obj = routes_dict.get(st.session_state.result.chosen_route_id)
            if chosen_route_obj:
                render_route_map([chosen_route_obj])
            
            st.subheader("Sensitivity Breakdown")
            render_sensitivity_chart(st.session_state.result, routes_dict, weights)

    with col2:
        if st.session_state.result:
            render_executive_summary(st.session_state.result, routes_dict)
            
            if st.session_state.critique:
                st.warning(f"**Self-Critique Auditor:** {st.session_state.critique}")
                
            st.subheader("Trade-off Matrix")
            render_tradeoff_matrix(st.session_state.result, selected_scenario["routes"], weights)
            render_rejection_panel(st.session_state.result, routes_dict)
            
            export_data = {
                "decision": st.session_state.result.dict(),
                "traces": [t.dict() for t in st.session_state.traces]
            }
            st.download_button("Download Audit Log (JSON)", data=json.dumps(export_data, indent=2), file_name="audit_log.json", mime="application/json")
            
            from ui_components import render_freight_quotes
            render_freight_quotes(st.session_state.result, routes_dict)
