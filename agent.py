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
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o", temperature=0)

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

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
        return " ".join(texts)
    return str(content)

def run_self_critique(result: TradeOffResult) -> str:
    llm = get_llm()
    prompt = f"A skeptical auditor is reviewing this decision to choose {result.chosen_route_id}. The justification given was: '{result.justification}'. Find one primary weakness or risk in this justification. Do not be overly harsh, just analytical. Output exactly one short paragraph."
    response = llm.invoke([HumanMessage(content=prompt)])
    return _extract_text(response.content)

def answer_followup(question: str, result: TradeOffResult, traces: List[ToolTrace]) -> str:
    llm = get_llm()
    trace_text = "\n".join([f"Tool {t.tool_name} returned {t.output}" for t in traces])
    prompt = f"Context Data:\n{trace_text}\n\nFinal Decision:\n{result.json()}\n\nQuestion from user: {question}\nAnswer the user directly and concisely using only the context provided."
    response = llm.invoke([HumanMessage(content=prompt)])
    return _extract_text(response.content)
