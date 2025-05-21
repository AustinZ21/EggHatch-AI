"""
Master Agent / Orchestrator for EggHatch AI.

This module implements the main orchestration logic for the EggHatch AI agent
using LangGraph for state management and flow control.
"""

import os
from typing import Dict, List, Optional, Union, Any
import json
import re
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from app.llm_integrations import OllamaClient
from app.agents.data_pipeline import get_data_pipeline
from app.agents.trend_analysis import TrendAnalysisAgent
from app.prompts import (
    MASTER_AGENT_SYSTEM_PROMPT,
    QUERY_UNDERSTANDING_PROMPT,
    RESPONSE_SYNTHESIS_PROMPT,
)

# Load environment variables
load_dotenv()

# Initialize LLM client
llm_client = OllamaClient()

# For the POC, we're focusing only on laptop recommendations with trend and sentiment analysis
DEFAULT_TASK_QUEUE = [
    "trend_and_sentiment_analysis"
]

# Task mapping based on query type
QUERY_TYPE_TASKS = {
    # For the POC, we're only focusing on laptop recommendations
    "laptop_recommendation": DEFAULT_TASK_QUEUE,
    "gaming_laptop": DEFAULT_TASK_QUEUE,
    "laptop_review": DEFAULT_TASK_QUEUE,
    "laptop_comparison": DEFAULT_TASK_QUEUE,
    "default": DEFAULT_TASK_QUEUE
}

# Initialize data pipeline
data_pipeline = get_data_pipeline()

# Define state schema
class AgentState(BaseModel):
    """State for the EggHatch AI agent."""
    
    # User interaction
    user_query: str = Field(default="")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    
    # Extracted information
    query_type: str = Field(default="")
    budget: Optional[str] = Field(default=None)
    use_case: Optional[str] = Field(default=None)
    requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Task management
    current_task: str = Field(default="")
    task_queue: List[str] = Field(default_factory=list)
    waiting_for: Optional[str] = Field(default=None) 
    
    # Scope and validation
    is_in_scope: bool = Field(default=True)
    
    # Tool results - POC focuses only on trend and sentiment analysis
    trend_insights: Optional[Dict[str, Any]] = Field(default=None)
    sentiment_analysis: Optional[Dict[str, Any]] = Field(default=None)
    
    # Response
    final_response: str = Field(default="")

# Define agent nodes

def initialize_state(state: AgentState) -> dict:
    """
    Initialize the agent state with the user query.
    This is required as the entry point for LangGraph to properly update the state.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state dictionary with explicitly modified fields
    """
    # LangGraph requires the entry point to return a dict with explicit changes
    # We need to return a dict with at least one field that's different from the input
    return {"user_query": state.user_query, "is_in_scope": True}
def understand_query(state: AgentState) -> AgentState:
    """
    Understand the user query and extract relevant information.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated agent state with extracted information
    """
    def parse_llm_response(response: str) -> bool:
        """Parse LLM response for YES/NO questions."""
        return response.strip().upper().startswith("YES")

    def extract_json_from_text(text: str) -> dict | None:
        """Extract JSON from text, handling code blocks."""
        try:
            # Try direct JSON parsing first
            return json.loads(text)
        except json.JSONDecodeError:
            # Look for JSON in code blocks
            blocks = text.split('```')
            for i in range(1, len(blocks), 2):
                if i < len(blocks) and 'json' in blocks[i-1].lower():
                    try:
                        return json.loads(blocks[i].strip())
                    except json.JSONDecodeError:
                        continue
            return None

    # Check if query is in scope
    scope_response = llm_client.generate(
        prompt=f"Determine if this query is about PC building, components, or tech gear. Answer YES/NO only.\nQuery: {state.user_query}",
        system_prompt="You are an AI that determines if queries are related to PC building or tech gear shopping."
    )
    
    if not parse_llm_response(scope_response.get("response", "")):
        state.is_in_scope = False
        state.final_response = "I'm specialized in PC building and tech gear shopping. What specific PC or tech-related information are you looking for today?"
        state.waiting_for = "in_scope_query"
        return state
    
    state.is_in_scope = True
    
    # Check if query is too vague
    vague_response = llm_client.generate(
        prompt=f"Determine if this query is too vague (e.g., 'I need a computer'). Answer YES/NO only.\nQuery: {state.user_query}",
        system_prompt="You are an AI that determines if tech queries are too vague to answer specifically."
    )
    
    if parse_llm_response(vague_response.get("response", "")):
        state.waiting_for = "query_details"
        state.final_response = "What kind of computer are you looking for? A laptop or a desktop? And what will you primarily be using it for?"
        return state
    
    # Process specific query
    response = llm_client.generate(
        prompt=QUERY_UNDERSTANDING_PROMPT.format(user_query=state.user_query),
        system_prompt=MASTER_AGENT_SYSTEM_PROMPT
    )
    
    # Extract information from response
    extracted_info = extract_json_from_text(response.get("response", ""))
    
    if not extracted_info:
        # Simple fallback extraction
        fallback_response = llm_client.generate(
            prompt=f"""
            Extract key information from this text as JSON:
            - query_type (e.g., gaming_pc_build, laptop_recommendation)
            - budget (or null)
            - use_case (e.g., gaming, content_creation)
            - requirements (dictionary)
            
            Text: {response.get("response", "")}
            
            Format: ```json
            {{
              "query_type": "example",
              "budget": "$1000",
              "use_case": "example",
              "requirements": {{}}
            }}
            ```
            """
        )
        extracted_info = extract_json_from_text(fallback_response.get("response", ""))
    
    # Update state with extracted information or defaults
    if extracted_info:
        state.query_type = extracted_info.get("query_type", "unknown")
        state.budget = extracted_info.get("budget")
        state.use_case = extracted_info.get("use_case")
        state.requirements = extracted_info.get("requirements", {})
    else:
        state.query_type = "unknown"
    
    return state

def extract_task_queue(response_text: str, state: AgentState) -> List[str]:
    """
    Extract the task queue from the LLM response.
    
    Args:
        response_text: The LLM response text
        state: Current agent state
        
    Returns:
        List of tasks to execute
    """
    if not state.query_type:
        return QUERY_TYPE_TASKS["default"]
    
    # Use exact query type match from LLM response
    return QUERY_TYPE_TASKS.get(state.query_type, QUERY_TYPE_TASKS["default"])


def decompose_tasks(state: AgentState) -> AgentState:
    """
    Set up the task queue based on the query type.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated agent state with task queue
    """
    # Get task queue from QUERY_TYPE_TASKS based on query_type
    state.task_queue = extract_task_queue("", state)
    
    # Set the current task to the first in queue
    state.current_task = state.task_queue[0] if state.task_queue else ""
    
    return state

def execute_current_task(state: AgentState) -> AgentState:
    """
    Execute the current task in the task queue - simplified for POC.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated agent state with task results
    """
    print(f"Executing task: {state.current_task}")
    
    try:
        if state.current_task == "trend_and_sentiment_analysis":
            print("Running Trend Analysis and Sentiment Analysis...")
            
            # Create a simple state object for the trend analysis agent
            class TrendState:
                def __init__(self):
                    self.query = state.user_query
                    self.trend_insights = None
                    self.sentiment_results = None
            
            # Get the sentiment analyzer for explicit use
            sentiment_analyzer = get_sentiment_analyzer()
            print(f"Initialized Sentiment Analyzer: {type(sentiment_analyzer).__name__}")
            
            # Run the trend analysis agent (which internally uses sentiment analysis)
            trend_state = TrendState()
            # Initialize the trend analysis agent
            trend_agent = TrendAnalysisAgent()
            # Run the analysis
            updated_trend_state = trend_agent.analyze_trends(trend_state.query)
            
            # Update the master agent state with trend and sentiment insights
            state.trend_insights = updated_trend_state.trend_insights
            
            # Store sentiment analysis results separately if available
            if hasattr(updated_trend_state, 'sentiment_results') and updated_trend_state.sentiment_results:
                state.sentiment_analysis = updated_trend_state.sentiment_results
            # If not available as a separate field, try to extract from trend_insights
            elif state.trend_insights and isinstance(state.trend_insights, dict) and "sentiment" in state.trend_insights:
                state.sentiment_analysis = {"results": state.trend_insights["sentiment"]}
            
            print("Trend Analysis and Sentiment Analysis completed successfully.")
            
            # Add explicit mention of sentiment analysis in the insights
            if state.trend_insights and isinstance(state.trend_insights, dict):
                state.trend_insights["sentiment_analysis_used"] = True
        
    except Exception as e:
        print(f"Error executing task {state.current_task}: {e}")
        # Simple error state
        if state.current_task == "trend_and_sentiment_analysis":
            state.trend_insights = {"analysis": "Error in trend and sentiment analysis."}
            state.sentiment_analysis = {"analysis": "Error in sentiment analysis."}

    # Update task queue
    if state.task_queue:
        state.task_queue.pop(0)
        state.current_task = state.task_queue[0] if state.task_queue else ""
    
    return state

def synthesize_response(state: AgentState) -> AgentState:
    """
    Synthesize a final response based on all collected information.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated agent state with final response
    """
    
    # Combine trend insights and sentiment analysis into a single comprehensive structure
    combined_insights = {}
    
    # Add trend insights if available
    if state.trend_insights:
        combined_insights.update(state.trend_insights)
    
    # Add sentiment analysis if available and not already included in trend insights
    if state.sentiment_analysis:
        # Check if sentiment is already in combined_insights to avoid duplication
        if "sentiment" not in combined_insights:
            combined_insights["sentiment"] = state.sentiment_analysis
    
    # Format the prompt with only the required parameters
    prompt = RESPONSE_SYNTHESIS_PROMPT.format(
        user_query=state.user_query,
        trend_insights=combined_insights
    )
    
    # Call the LLM
    response = llm_client.generate(
        prompt=prompt,
        system_prompt=MASTER_AGENT_SYSTEM_PROMPT
    )
    
    # Extract the actual response from the LLM output
    response_text = response.get("response", "")
    
    # Use the response if it's not empty, otherwise provide a fallback
    if response_text.strip():
        state.final_response = response_text.strip()
    else:
        # Fallback in case the LLM returns an empty response
        print("Warning: LLM returned empty response in synthesize_response. Using fallback.")
        state.final_response = f"I've analyzed your request about {state.query_type or 'tech gear'}. "
        
        if state.trend_insights:
            state.final_response += "Based on trend and sentiment analysis, here are my insights: "
            if isinstance(state.trend_insights, dict):
                # Add trend insights
                if "topics" in state.trend_insights:
                    state.final_response += f"\n\n**Popular Topics:**\n"
                    for topic in state.trend_insights["topics"]:
                        state.final_response += f"- {topic}\n"
                
                # Add sentiment summary if available
                if "sentiment_summary" in state.trend_insights:
                    state.final_response += f"\n**Sentiment Analysis:**\n{state.trend_insights['sentiment_summary']}\n"
                
                # Add recommendations if available
                if "recommendations" in state.trend_insights:
                    state.final_response += f"\n**Recommendations:**\n{state.trend_insights['recommendations']}\n"
        else:
            state.final_response += "I don't have enough information to provide specific insights at this time. Could you provide more details about what you're looking for?"
    
    # Update conversation history
    state.conversation_history.append({
        "role": "user",
        "content": state.user_query
    })
    state.conversation_history.append({
        "role": "assistant",
        "content": state.final_response
    })
    
    # Fix the f-string syntax
    if len(state.user_query) > 50:
        print(f"Generated response for query: '{state.user_query[:50]}...'")
    else:
        print(f"Generated response for query: '{state.user_query}'")
    print(f"Response length: {len(state.final_response)} characters")
    
    # For debugging
    if len(state.final_response) < 100:
        print(f"Warning: Short response generated: '{state.final_response}'")
    elif len(state.final_response) > 2000:
        print(f"Warning: Very long response generated: {len(state.final_response)} characters")

    
    return state

def ask_for_budget(state: AgentState) -> AgentState:
    """
    Ask the user for their budget.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated agent state with a question about budget
    """
    state.final_response = "To provide you with the best recommendations, could you please let me know what your budget is for this purchase?"
    state.waiting_for = "budget"
    return state

def ask_for_use_case(state: AgentState) -> AgentState:
    """
    Ask the user for their intended use case.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated agent state with a question about use case
    """
    state.final_response = "To help you find the right option, could you tell me what you'll primarily be using this for? (e.g., gaming, content creation, office work)"
    state.waiting_for = "use_case"
    return state

def check_information_completeness(state: AgentState) -> str:
    """
    Check if all necessary information is available to process the query.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node to execute
    """
    # If query is out of scope, don't proceed with task decomposition
    if not state.is_in_scope:
        return "out_of_scope_response"
    
    # For PC build requests, budget is critical
    if state.query_type in ["gaming_pc_build", "pc_build", "build_recommendation"] and not state.budget:
        return "ask_for_budget"
    
    # For laptop recommendations, both budget and use case are important
    elif state.query_type in ["laptop_recommendation", "laptop_suggestion"]:
        if not state.budget:
            return "ask_for_budget"
        elif not state.use_case:
            return "ask_for_use_case"
    
    # If we have all necessary information, proceed to task decomposition
    return "decompose_tasks"

def out_of_scope_response(state: AgentState) -> AgentState:
    """
    Handle out-of-scope queries.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated agent state with out-of-scope response
    """
    # Final response is already set in understand_query
    return state

def should_continue_tasks(state: AgentState) -> str:
    """
    Determine whether to continue executing tasks or synthesize a response.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node to execute
    """
    if state.task_queue:
        return "execute_current_task"
    else:
        return "synthesize_response"

# Create the graph
def create_agent_graph() -> StateGraph:
    """
    Create the agent graph for orchestrating the EggHatch AI workflow.
    
    Returns:
        StateGraph instance
    """
    # Initialize the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("initialize_state", initialize_state)
    graph.add_node("understand_query", understand_query)
    graph.add_node("decompose_tasks", decompose_tasks)
    graph.add_node("execute_current_task", execute_current_task)
    graph.add_node("synthesize_response", synthesize_response)
    graph.add_node("ask_for_budget", ask_for_budget)
    graph.add_node("ask_for_use_case", ask_for_use_case)
    graph.add_node("out_of_scope_response", out_of_scope_response)
    
    # Add conditional edge from understand_query based on information completeness
    graph.add_conditional_edges(
        "understand_query",
        check_information_completeness,
        {
            "decompose_tasks": "decompose_tasks",
            "ask_for_budget": "ask_for_budget",
            "ask_for_use_case": "ask_for_use_case",
            "out_of_scope_response": "out_of_scope_response"
        }
    )
    
    # Add edges for task execution
    graph.add_edge("decompose_tasks", "execute_current_task")
    graph.add_conditional_edges(
        "execute_current_task",
        should_continue_tasks,
        {
            "execute_current_task": "execute_current_task",
            "synthesize_response": "synthesize_response"
        }
    )
    
    # Add edges to END
    graph.add_edge("synthesize_response", END)
    graph.add_edge("ask_for_budget", END)
    graph.add_edge("ask_for_use_case", END)
    graph.add_edge("out_of_scope_response", END)
    
    # Add edge from initialize_state to understand_query
    graph.add_edge("initialize_state", "understand_query")
    
    # Set the entry point
    graph.set_entry_point("initialize_state")
    
    return graph

# Create the agent
agent_graph = create_agent_graph()
agent = agent_graph.compile()

def process_query(query: str) -> str:
    """
    Process a user query through the EggHatch AI agent.
    
    Args:
        query: User query string
        
    Returns:
        Agent response
    """
    # Initialize state as a dictionary for LangGraph's invoke method with all required fields
    initial_input_dict = {
        "user_query": query,
        "conversation_history": [],
        "query_type": "",
        "budget": None,
        "use_case": None,
        "requirements": {},
        "current_task": "",
        "task_queue": [],
        "waiting_for": None,
        "is_in_scope": True,
        "trend_insights": None,
        "sentiment_analysis": None,
        "final_response": ""
    }
    
    # Run the agent
    final_state_dict = agent.invoke(initial_input_dict)
    
    # Return the final response from the state dictionary
    return final_state_dict.get("final_response", "Error: Agent did not produce a final response")

# Example usage
if __name__ == "__main__":
    query = "I want to build a gaming PC for about $2000 that can run Cyberpunk 2077 with ray tracing."
    response = process_query(query)
    print(response)
