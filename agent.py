import time
from typing import List, Dict, Any, Callable
import os
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from models import RouteOption, TradeOffResult, ToolTrace
from tools import get_route_cost, get_route_time, get_route_risk, get_route_carbon

def get_llm():
    if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-72B-Instruct",
            task="text-generation",
            max_new_tokens=1024,
            do_sample=False,
        )
        return ChatHuggingFace(llm=endpoint)
    elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    
    # Fallback to UncloseAI free endpoint
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="Lorbus/Qwen3.6-27B-int4-AutoRound",
        api_key="empty",
        base_url="https://hermes.ai.unturf.com/v1",
        temperature=0
    )

def run_agent(scenario: Dict[str, Any], routes: List[RouteOption], trace_callback: Callable[[ToolTrace], None]) -> TradeOffResult:
    llm = get_llm()
    tools = [get_route_cost, get_route_time, get_route_risk, get_route_carbon]
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = (
        "You are an autonomous supply chain orchestrator. "
        "Your task is to evaluate a list of candidate routes for a given disruption scenario. "
        "You must use the provided tools to find the cost, time, risk, and carbon scores for EVERY route. "
        "You must call the tools autonomously. "
        "Once you have all data, you must stop calling tools."
    )
    
    routes_info = "\n".join([f"- Route ID: {r.id}, Name: {r.name}, Mode: {r.mode}, Route: {r.origin} to {r.destination}" for r in routes])
    human_prompt = f"Scenario: {scenario['name']}\nDescription: {scenario['description']}\n\nCandidate Routes:\n{routes_info}"
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    
    tools_by_name = {t.name: t for t in tools}
    
    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            break
            
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            tool_func = tools_by_name[tool_name]
            tool_output = tool_func.invoke(tool_args)
            
            trace_callback(ToolTrace(
                tool_name=tool_name,
                input_args=tool_args,
                output=tool_output,
                timestamp=str(time.time())
            ))
            
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
    final_messages = messages + [HumanMessage(content="Now output the final decision using the TradeOffResult schema. Include exactly one chosen route and all others in rejected routes. Include the scores for all evaluated routes.")]
    
    # Check if the LLM supports structured output
    try:
        llm.with_structured_output(TradeOffResult)
        supports_structured = True
    except NotImplementedError:
        supports_structured = False
        
    if supports_structured:
        llm_structured = llm.with_structured_output(TradeOffResult)
        try:
            return llm_structured.invoke(final_messages)
        except Exception as e:
            retry_messages = final_messages + [HumanMessage(content=f"Validation failed: {str(e)}. Please correct your output and return the valid TradeOffResult.")]
            return llm_structured.invoke(retry_messages)
    else:
        from langchain_core.output_parsers import JsonOutputParser
        import json
        
        parser = JsonOutputParser(pydantic_object=TradeOffResult)
        format_instructions = parser.get_format_instructions()
        
        fallback_msg = HumanMessage(content=f"Now output the final decision in JSON format. Include exactly one chosen route and all others in rejected routes. Include the scores for all evaluated routes.\n\n{format_instructions}\n\nReturn ONLY raw, valid JSON. Do not include markdown formatting or conversational text.")
        fallback_messages = messages + [fallback_msg]
        
        for attempt in range(3):
            try:
                response = llm.invoke(fallback_messages)
                content = _extract_text(response.content)
                
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                    
                parsed = json.loads(content)
                return TradeOffResult(**parsed)
            except Exception as e:
                if attempt == 2:
                    # Last resort, return a dummy fallback to prevent crashing the UI completely
                    return TradeOffResult(chosen_route_id=routes[0].id, justification=f"Failed to parse AI output after multiple attempts. Error: {str(e)}", route_scores={}, rejected_routes=[])
                fallback_messages.append(response)
                fallback_messages.append(HumanMessage(content=f"Validation failed: {str(e)}. Please output ONLY valid JSON."))

import re

def _extract_text(content) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
        text = " ".join(texts)
    else:
        text = str(content)
    # Robustly strip thinking blocks even if the opening <think> tag is missing
    if "</think>" in text:
        text = text.split("</think>")[-1]
    
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

def run_self_critique(result: TradeOffResult, routes_dict: Dict[str, RouteOption] = None) -> str:
    """Generate a data-driven self-critique by analyzing actual route scores.
    This avoids the local LLM entirely for reliability and speed."""
    
    chosen_id = result.chosen_route_id
    scores = result.route_scores
    
    # If we have no score data, return a generic but still useful critique
    if not scores or chosen_id not in scores:
        return (
            "Without transparent scoring data, it is impossible to verify whether this route "
            "was truly optimal or if the decision was influenced by incomplete information. "
            "The absence of comparative metrics is itself a significant audit risk."
        )
    
    chosen_scores = scores[chosen_id]
    other_routes = {rid: s for rid, s in scores.items() if rid != chosen_id}
    
    if not other_routes:
        return (
            "Only one route was evaluated, making it impossible to confirm this was the best option. "
            "A robust supply chain strategy requires evaluating multiple alternatives to avoid single-point-of-failure decisions."
        )
    
    critiques = []
    
    # Mode-specific operational risk insights
    mode_risks = {
        "air": "Air freight, while fast, is highly exposed to fuel price volatility, airport congestion, and capacity crunches during peak seasons, making the projected cost far more volatile than the justification suggests.",
        "sea": "Maritime shipping faces significant risks from port congestion, container shortages, and weather-induced delays that can extend transit times by weeks, undermining the reliability assumed in this justification.",
        "rail": "Overland rail freight through multiple international borders introduces significant risks of customs bottlenecks, regulatory delays, and infrastructure disruptions that can easily derail the projected timeline.",
        "road": "Overland trucking across multiple international borders introduces significant risks of customs bottlenecks, regulatory delays, and regional instability that can easily disrupt schedules and inflate costs, making the projected timeline and pricing far more volatile than the justification suggests.",
        "sea-air": "The sea-air hybrid approach introduces a critical handoff point where cargo must be transshipped between modes, creating vulnerability to port delays, documentation mismatches, and coordination failures that compound transit time uncertainty.",
        "sea-rail": "The sea-rail combination requires seamless intermodal transfers that are often disrupted by port-rail connectivity issues, gauge changes at borders, and misaligned scheduling between maritime and rail operators."
    }
    
    # Check if chosen route has the HIGHEST cost
    min_cost_route = min(other_routes.items(), key=lambda x: x[1].cost)
    if min_cost_route[1].cost < chosen_scores.cost * 0.7:
        savings_pct = round((1 - min_cost_route[1].cost / chosen_scores.cost) * 100)
        critiques.append(
            f"The justification overlooks that alternative route {min_cost_route[0]} offers approximately "
            f"{savings_pct}% lower cost, raising serious questions about whether speed was overweighted "
            f"at the expense of capital efficiency in the current evaluation."
        )
    
    # Check if chosen route has high risk relative to alternatives
    min_risk_route = min(other_routes.items(), key=lambda x: x[1].risk)
    if chosen_scores.risk > min_risk_route[1].risk * 1.3:
        critiques.append(
            f"The chosen route carries a significantly elevated risk score compared to {min_risk_route[0]}, "
            f"suggesting the decision underestimates operational disruption potential. In volatile geopolitical "
            f"corridors, this risk differential could translate to costly delays and insurance premium spikes."
        )
    
    # Check carbon footprint concerns
    min_carbon_route = min(other_routes.items(), key=lambda x: x[1].carbon)
    if chosen_scores.carbon > min_carbon_route[1].carbon * 2.0:
        critiques.append(
            f"The carbon footprint of the chosen route is more than double that of {min_carbon_route[0]}, "
            f"which poses a material ESG compliance risk and could trigger carbon tax penalties under "
            f"tightening emissions regulations in major trade corridors."
        )
    
    # Check if a faster alternative exists but wasn't chosen
    min_time_route = min(other_routes.items(), key=lambda x: x[1].time)
    if min_time_route[1].time < chosen_scores.time * 0.6:
        critiques.append(
            f"A significantly faster alternative ({min_time_route[0]}) was rejected despite offering "
            f"substantially shorter transit time, which in time-sensitive disruption scenarios could "
            f"mean the difference between supply continuity and production line shutdowns."
        )
    
    # Pick the most impactful critique, or fall back to mode-specific risk
    if critiques:
        return critiques[0]
    
    # Fall back to mode-specific operational risk analysis
    chosen_mode = None
    if routes_dict and chosen_id in routes_dict:
        chosen_mode = routes_dict[chosen_id].mode
    else:
        # Try to determine mode from route_id patterns
        for mode_key in mode_risks:
            if mode_key in chosen_id.lower():
                chosen_mode = mode_key
                break
    
    if chosen_mode and chosen_mode in mode_risks:
        return mode_risks[chosen_mode]
    
    return (
        "While the chosen route may appear optimal on aggregate metrics, the justification does not "
        "adequately address scenario-specific vulnerabilities such as single-carrier dependency, "
        "corridor congestion during peak disruption periods, or the cascading impact of secondary delays "
        "on downstream supply chain commitments."
    )

def answer_followup(question: str, result: TradeOffResult, traces: List[ToolTrace]) -> str:
    llm = get_llm()
    trace_text = "\n".join([f"Tool {t.tool_name} returned {t.output}" for t in traces])
    prompt = f"Context Data:\n{trace_text}\n\nFinal Decision:\n{result.json()}\n\nQuestion from user: {question}\nAnswer the user directly and concisely using only the context provided."
    response = llm.invoke([HumanMessage(content=prompt)])
    return _extract_text(response.content)
