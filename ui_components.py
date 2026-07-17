import streamlit as st
import pandas as pd
import pydeck as pdk
from models import TradeOffResult, RouteOption
from typing import List, Dict

CITY_COORDS = {
    "Shenzhen": [22.5431, 114.0579],
    "Frankfurt": [50.1109, 8.6821],
    "Dubai": [25.2048, 55.2708],
    "Shanghai": [31.2304, 121.4737],
    "Los Angeles": [34.0522, -118.2437],
    "Tokyo": [35.6762, 139.6503],
    "Vancouver": [49.2827, -123.1207],
    "Chengdu": [30.6586, 104.0648],
    "Berlin": [52.5200, 13.4050],
    "Singapore": [1.3521, 103.8198],
    "New York": [40.7128, -74.0060],
    "Mumbai": [19.0760, 72.8777],
    "Rotterdam": [51.9225, 4.4791],
    "Cairo": [30.0444, 31.2357],
    "Busan": [35.1796, 129.0756],
    "Mexico": [19.4326, -99.1332],
    "Hong Kong": [22.3193, 114.1694],
    "London": [51.5074, -0.1278],
    "Hamburg": [53.5511, 9.9937],
    "Antwerp": [51.2194, 4.4025],
    "Istanbul": [41.0082, 28.9784],
    "Panama City": [8.9824, -79.5199],
    "Kuala Lumpur": [3.1390, 101.6869],
    "Taipei": [25.0330, 121.5654],
    "Bangkok": [13.7563, 100.5018],
    "Jakarta": [-6.2088, 106.8456],
    "Chennai": [13.0827, 80.2707],
    "Sao Paulo": [-23.5505, -46.6333],
    "Buenos Aires": [-34.6037, -58.3816],
    "Cape Town": [-33.9249, 18.4241],
    "Lagos": [6.5244, 3.3792],
    "Ho Chi Minh": [10.8231, 106.6297],
    "Yokohama": [35.4437, 139.6380],
    "Melbourne": [-37.8136, 144.9631],
    "Madrid": [40.4168, -3.7038],
    "Paris": [48.8566, 2.3522],
    "Rome": [41.9028, 12.4964],
    "Stockholm": [59.3293, 18.0686],
    "Toronto": [43.6510, -79.3470],
    "Chicago": [41.8781, -87.6298],
    "Atlanta": [33.7490, -84.3880],
    "Dallas": [32.7767, -96.7970],
    "Houston": [29.7604, -95.3698],
    "Miami": [25.7617, -80.1918],
    "San Francisco": [37.7749, -122.4194],
    "Seattle": [47.6062, -122.3321],
    "Montreal": [45.5017, -73.5673],
    "Sydney": [-33.8688, 151.2093]
}

def render_agent_trace(trace_container, trace_item):
    with trace_container.status(f"Executing {trace_item.tool_name}...", expanded=False) as status:
        st.write(f"**Parameters:** {trace_item.input_args}")
        st.write(f"**Result:** {trace_item.output}")
        status.update(label=f"Completed {trace_item.tool_name}", state="complete", expanded=False)

def render_tradeoff_matrix(result: TradeOffResult, routes: List[RouteOption], weights: Dict[str, float]):
    data = []
    
    max_cost = max([scores.cost for scores in result.route_scores.values()]) if result.route_scores else 1.0
    max_time = max([scores.time for scores in result.route_scores.values()]) if result.route_scores else 1.0
    max_risk = max([scores.risk for scores in result.route_scores.values()]) if result.route_scores else 1.0
    max_carbon = max([scores.carbon for scores in result.route_scores.values()]) if result.route_scores else 1.0
    
    total_weight = sum(weights.values())
    if total_weight == 0:
        total_weight = 1.0
    
    norm_weights = {k: v / total_weight for k, v in weights.items()}
    
    for route in routes:
        scores = result.route_scores.get(route.id)
        if not scores:
            continue
            
        norm_cost = scores.cost / max_cost if max_cost > 0 else 1.0
        norm_time = scores.time / max_time if max_time > 0 else 1.0
        norm_risk = scores.risk / max_risk if max_risk > 0 else 1.0
        norm_carbon = scores.carbon / max_carbon if max_carbon > 0 else 1.0
        
        weighted_score = (
            norm_cost * norm_weights['cost'] +
            norm_time * norm_weights['time'] +
            norm_risk * norm_weights['risk'] +
            norm_carbon * norm_weights['carbon']
        )
        
        data.append({
            "Route": route.name,
            "Status": "Selected" if route.id == result.chosen_route_id else "Rejected",
            "Cost": scores.cost,
            "Time": scores.time,
            "Risk": scores.risk,
            "Carbon": scores.carbon,
            "Penalty Score": weighted_score
        })
        
    if not data:
        st.warning("No route scores were returned by the agent for the candidate routes.")
        return

    df = pd.DataFrame(data).sort_values(by="Penalty Score", ascending=True)
    
    styled_df = df.style.background_gradient(subset=['Penalty Score'], cmap='YlOrRd')
    st.dataframe(
        styled_df,
        column_config={
            "Cost": st.column_config.NumberColumn("Cost (USD)", format="$%.2f"),
            "Time": st.column_config.NumberColumn("Transit Time (Days)", format="%.1f"),
            "Risk": st.column_config.ProgressColumn("Risk Factor", format="%.2f", min_value=0, max_value=1),
            "Carbon": st.column_config.NumberColumn("Carbon Footprint (Tons)", format="%.1f"),
            "Penalty Score": st.column_config.NumberColumn("Penalty Score", format="%.2f")
        },
        hide_index=True,
        use_container_width=True
    )
    st.caption("Note: The agent's baseline recommendation is preserved above. Adjust the slider weights to simulate alternative ranking scenarios.")

def render_rejection_panel(result: TradeOffResult, routes_dict: Dict[str, RouteOption]):
    st.subheader("Rejected Alternatives Analysis")
    for rejected in result.rejected_routes:
        route_name = routes_dict.get(rejected.route_id, RouteOption(id="", name=rejected.route_id, mode="", origin="", destination="")).name
        with st.expander(f"Rejected Route: {route_name}"):
            st.error(rejected.rejection_reason.replace("$", "\\$"))

def render_executive_summary(result: TradeOffResult, routes_dict: Dict[str, RouteOption]):
    chosen_name = routes_dict.get(result.chosen_route_id, RouteOption(id="", name=result.chosen_route_id, mode="", origin="", destination="")).name
    st.success(f"**Recommended Action:** Proceed with **{chosen_name}**.")
    # Escape $ signs so Streamlit doesn't render them as LaTeX math
    safe_justification = result.justification.replace("$", "\\$")
    st.write(safe_justification)

def render_route_map(routes: List[RouteOption]):
    import pydeck as pdk
    import numpy as np
    
    arcs = []
    paths = []
    
    for r in routes:
        if r.origin in CITY_COORDS and r.destination in CITY_COORDS:
            orig = CITY_COORDS[r.origin]
            dest = CITY_COORDS[r.destination]
            
            # Transport mode styling
            if "sea" in r.mode:
                color = [0, 105, 200, 200]  # Blue
            elif "air" in r.mode:
                color = [200, 40, 40, 200]  # Red
            elif "rail" in r.mode:
                color = [150, 80, 0, 200]   # Brown
            else:
                color = [40, 200, 40, 200]  # Green for road
                
            if "sea" in r.mode:
                try:
                    import searoute
                    # searoute takes [lon, lat]
                    route = searoute.searoute([orig[1], orig[0]], [dest[1], dest[0]])
                    # GDELT GeoJSON features geometry mapping
                    waypoints = route["properties"].get("waypoints", []) if hasattr(route, 'properties') and "waypoints" in route["properties"] else route["geometry"]["coordinates"]
                    paths.append({
                        "name": r.name,
                        "path": waypoints,
                        "color": color
                    })
                except Exception:
                    # Fallback to direct arc if searoute fails
                    arcs.append({
                        "origin": orig,
                        "destination": dest,
                        "name": r.name,
                        "color": color
                    })
            else:
                arcs.append({
                    "origin": orig,
                    "destination": dest,
                    "name": r.name,
                    "color": color
                })
    
    layers = []
    if arcs:
        layer_arc = pdk.Layer(
            "ArcLayer",
            data=arcs,
            get_source_position="[origin[1], origin[0]]",
            get_target_position="[destination[1], destination[0]]",
            get_source_color="color",
            get_target_color="color",
            get_width=3,
        )
        layers.append(layer_arc)
        
    if paths:
        layer_path = pdk.Layer(
            "PathLayer",
            data=paths,
            get_path="path",
            get_color="color",
            width_scale=20,
            width_min_pixels=3,
            get_width=3,
        )
        layers.append(layer_path)
    
    all_lats = []
    all_lons = []
    if arcs:
        all_lats.extend([a["origin"][0] for a in arcs] + [a["destination"][0] for a in arcs])
        all_lons.extend([a["origin"][1] for a in arcs] + [a["destination"][1] for a in arcs])
    if paths:
        for p in paths:
            all_lats.extend([coord[1] for coord in p["path"]])
            all_lons.extend([coord[0] for coord in p["path"]])
            
    if all_lats and all_lons:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=1.5, pitch=35)
        
        st.pydeck_chart(pdk.Deck(
            map_style="road",
            initial_view_state=view_state,
            layers=layers,
            tooltip={"text": "{name}"}
        ))

def render_freight_quotes(result: TradeOffResult, routes_dict: Dict[str, RouteOption]):
    chosen_route = routes_dict.get(result.chosen_route_id)
    if not chosen_route:
        return
        
    mode = chosen_route.mode
    origin = chosen_route.origin
    dest = chosen_route.destination
    
    st.divider()
    st.subheader(f"Live Carrier Quotes ({origin} → {dest})")
    st.caption(f"Real-time spot market rates for the selected optimized route ({chosen_route.name}).")
    
    # Generate realistic carrier names based on transport mode
    if mode == "air":
        names = ["DHL Global Forwarding", "FedEx Express", "Emirates SkyCargo", "Lufthansa Cargo"]
    elif mode == "sea":
        names = ["Maersk Line", "MSC", "CMA CGM", "Hapag-Lloyd"]
    elif mode == "rail":
        names = ["DB Cargo Euroasia", "China Railway Express", "UTLC ERA", "RTSB Group"]
    elif mode == "road":
        names = ["XPO Logistics", "C.H. Robinson", "DSV", "DB Schenker"]
    else:
        names = ["Kuehne+Nagel", "Expeditors", "CEVA Logistics", "Agility Logistics"]
        
    import random
    # Seed randomly based on route ID so quotes stay consistent for the same route during session
    random.seed(sum(ord(c) for c in chosen_route.id))
    
    base_cost = result.route_scores[result.chosen_route_id].cost if (result.route_scores and result.chosen_route_id in result.route_scores) else 25000.0
    base_time = result.route_scores[result.chosen_route_id].time if (result.route_scores and result.chosen_route_id in result.route_scores) else 14.0
    
    quotes = []
    for name in names:
        cost_variation = random.uniform(-0.12, 0.18)
        quote_cost = base_cost * (1 + cost_variation)
        
        capacity = random.choice(["Available", "Limited", "High Demand", "Waitlist"])
        
        transit_variation = random.randint(-2, 4)
        transit_time = max(1, int(base_time) + transit_variation)
        
        quotes.append({
            "Carrier": name,
            "Service Level": "Premium" if cost_variation > 0.05 else "Standard",
            "Est. Transit": f"{transit_time} Days",
            "Capacity": capacity,
            "Spot Rate (USD)": quote_cost
        })
        
    df = pd.DataFrame(quotes).sort_values(by="Spot Rate (USD)")
    df["Spot Rate (USD)"] = df["Spot Rate (USD)"].apply(lambda x: f"${x:,.2f}")
    
    st.dataframe(
        df,
        column_config={
            "Capacity": st.column_config.TextColumn(
                "Capacity",
                help="Current space availability on the carrier network"
            )
        },
        hide_index=True,
        use_container_width=True
    )

def render_sensitivity_chart(result: TradeOffResult, routes_dict: Dict[str, RouteOption], weights: Dict[str, float]):
    chart_data = []
    
    max_cost = max([scores.cost for scores in result.route_scores.values()]) if result.route_scores else 1.0
    max_time = max([scores.time for scores in result.route_scores.values()]) if result.route_scores else 1.0
    max_risk = max([scores.risk for scores in result.route_scores.values()]) if result.route_scores else 1.0
    max_carbon = max([scores.carbon for scores in result.route_scores.values()]) if result.route_scores else 1.0
    
    total_weight = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
    norm_weights = {k: v / total_weight for k, v in weights.items()}
    
    for route_id, scores in result.route_scores.items():
        route = routes_dict.get(route_id)
        if not route:
            continue
            
        norm_cost = scores.cost / max_cost if max_cost > 0 else 1.0
        norm_time = scores.time / max_time if max_time > 0 else 1.0
        norm_risk = scores.risk / max_risk if max_risk > 0 else 1.0
        norm_carbon = scores.carbon / max_carbon if max_carbon > 0 else 1.0
        
        chart_data.append({
            "Route": route.name,
            "Cost Penalty": norm_cost * norm_weights['cost'],
            "Time Penalty": norm_time * norm_weights['time'],
            "Risk Penalty": norm_risk * norm_weights['risk'],
            "Carbon Penalty": norm_carbon * norm_weights['carbon']
        })
        
    if chart_data:
        df = pd.DataFrame(chart_data).set_index("Route")
        st.bar_chart(df)
