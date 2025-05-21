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
from app.agents.sentiment_analysis import get_sentiment_analyzer
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
def collect_streaming_response(generator):
    """
    Collect a complete response from a streaming generator.
    
    Args:
        generator: Generator yielding response chunks
        
    Returns:
        Dictionary with the complete response
    """
    full_response = ""
    for chunk in generator:
        if "response" in chunk:
            full_response += chunk["response"]
    
    # Return in the same format as non-streaming responses
    return {"response": full_response}

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
        response = response.strip().upper()
        # Check for various forms of YES responses
        return "YES" in response or response.startswith("Y") or "CORRECT" in response or "TRUE" in response

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
    scope_generator = llm_client.generate(
        prompt=f"Determine if this query is about PC building, components, tech gear, or gaming laptops. Answer YES/NO only.\nQuery: {state.user_query}",
        system_prompt="You are an AI that determines if queries are related to PC building, tech gear shopping, or gaming laptops. Queries about gaming laptops are DEFINITELY in scope."
    )
    scope_response = collect_streaming_response(scope_generator)
    
    if not parse_llm_response(scope_response.get("response", "")):
        # Return a dictionary with only the changed fields
        return {
            "is_in_scope": False,
            "final_response": "I'm specialized in PC building and tech gear shopping. What specific PC or tech-related information are you looking for today?",
            "waiting_for": "in_scope_query"
        }
    
    state.is_in_scope = True
    
    # Check if query is too vague
    vague_generator = llm_client.generate(
        prompt=f"Determine if this query is too vague (e.g., 'I need a computer'). Answer YES/NO only.\nQuery: {state.user_query}",
        system_prompt="You are an AI that determines if tech queries are too vague to answer specifically."
    )
    vague_response = collect_streaming_response(vague_generator)
    
    if parse_llm_response(vague_response.get("response", "")):
        # Return a dictionary with only the changed fields
        return {
            "waiting_for": "query_details",
            "final_response": "What kind of computer are you looking for? A laptop or a desktop? And what will you primarily be using it for?"
        }
    
    # Process specific query
    response_generator = llm_client.generate(
        prompt=QUERY_UNDERSTANDING_PROMPT.format(user_query=state.user_query),
        system_prompt=MASTER_AGENT_SYSTEM_PROMPT
    )
    response = collect_streaming_response(response_generator)
    
    # Extract information from response
    extracted_info = extract_json_from_text(response.get("response", ""))
    
    if not extracted_info:
        # Simple fallback extraction
        fallback_generator = llm_client.generate(
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
        fallback_response = collect_streaming_response(fallback_generator)
        extracted_info = extract_json_from_text(fallback_response.get("response", ""))
    
    # Create a dictionary with only the changed fields
    updated_fields = {}
    
    # Handle out-of-scope queries
    if not state.is_in_scope:
        updated_fields["is_in_scope"] = False
        updated_fields["waiting_for"] = "in_scope_query"
        updated_fields["final_response"] = "I'm specialized in PC building and tech gear shopping. What specific PC or tech-related information are you looking for today?"
        return updated_fields
    
    # Handle too vague queries
    if state.waiting_for == "query_details":
        updated_fields["waiting_for"] = "query_details"
        updated_fields["final_response"] = "What kind of computer are you looking for? A laptop or a desktop? And what will you primarily be using it for?"
        return updated_fields
    
    # Update fields based on extracted information
    updated_fields["is_in_scope"] = True
    
    if extracted_info:
        updated_fields["query_type"] = extracted_info.get("query_type", "unknown")
        updated_fields["budget"] = extracted_info.get("budget")
        updated_fields["use_case"] = extracted_info.get("use_case")
        updated_fields["requirements"] = extracted_info.get("requirements", {})
    else:
        updated_fields["query_type"] = "unknown"
    
    return updated_fields

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


def decompose_tasks(state) -> dict:
    """
    Set up the task queue based on the query type.
    
    Args:
        state: Current agent state (can be dictionary or AgentState)
        
    Returns:
        Dictionary with updated fields
    """
    # Handle both dictionary and AgentState objects
    if hasattr(state, 'get'):
        # It's a dictionary
        query_type = state.get("query_type", "").lower()
    else:
        # It's a Pydantic model (AgentState)
        query_type = state.query_type.lower() if state.query_type else ""
    
    # Get task queue based on query type or use default
    task_queue = QUERY_TYPE_TASKS.get(query_type, QUERY_TYPE_TASKS["default"])
    
    # Return only the updated fields
    return {"task_queue": task_queue}

def execute_current_task(state) -> dict:
    """
    Execute the current task in the task queue - simplified for POC.
    
    Args:
        state: Current agent state (can be dictionary or AgentState)
        
    Returns:
        Dictionary with updated fields
    """
    # Handle both dictionary and AgentState objects
    if hasattr(state, 'get'):
        # It's a dictionary
        user_query = state.get("user_query", "")
        task_queue = state.get("task_queue", [])
    else:
        # It's a Pydantic model (AgentState)
        user_query = state.user_query
        task_queue = state.task_queue
    
    # Get the current task from the task queue if available
    current_task = ""
    if task_queue and len(task_queue) > 0:
        current_task = task_queue[0]  # Get the first task in the queue
        
    print(f"Executing task: {current_task}")
    
    # Initialize the dictionary for updated fields
    updated_fields = {}
    
    # Remove the current task from the queue for the next iteration
    if task_queue and len(task_queue) > 0:
        updated_task_queue = task_queue[1:]
        updated_fields["task_queue"] = updated_task_queue
        print(f"[MASTER AGENT] Updated task queue: {updated_task_queue}")
    
    try:
        print(f"[MASTER AGENT] Task queue: {task_queue}")
        print(f"[MASTER AGENT] Current task: {current_task}")
        
        if current_task == "trend_and_sentiment_analysis":
            print("[MASTER AGENT] Running Trend Analysis and Sentiment Analysis...")
            
            # Create a simple state object for the trend analysis agent
            class TrendState:
                def __init__(self):
                    # Use the user_query we already extracted above
                    self.query = user_query
                    self.trend_insights = None
                    self.sentiment_results = None
            
            # Get the sentiment analyzer for explicit use
            sentiment_analyzer = get_sentiment_analyzer()
            print(f"Initialized Sentiment Analyzer: {type(sentiment_analyzer).__name__}")
            
            # Run the trend analysis agent (which internally uses sentiment analysis)
            print(f"[MASTER AGENT] Starting trend analysis for query: '{user_query}'")
            # Initialize the trend analysis agent
            trend_agent = TrendAnalysisAgent()
            # Run the analysis - this returns a dictionary, not an updated state
            trend_results = trend_agent.analyze_trends(user_query)
            
            # Log the results
            print(f"[MASTER AGENT] Completed trend analysis, found {len(trend_results.get('top_laptops', []))} laptop recommendations")
            
            # Update the fields with trend insights
            updated_fields["trend_insights"] = trend_results
            print(f"[MASTER AGENT] Trend insights keys: {trend_results.keys() if trend_results else 'None'}")
            
            # Debug logging
            if not trend_results:
                print("[MASTER AGENT] WARNING: Trend analysis returned empty results")
            elif not trend_results.get('top_laptops'):
                print("[MASTER AGENT] WARNING: No top laptops found in trend analysis results")
                print(f"[MASTER AGENT] Available keys in trend results: {trend_results.keys()}")
            else:
                print(f"[MASTER AGENT] Successfully found {len(trend_results.get('top_laptops', []))} laptop recommendations")
                for i, laptop in enumerate(trend_results.get('top_laptops', [])[:2]):
                    print(f"[MASTER AGENT] Top laptop {i+1}: {laptop.get('name', 'Unknown')} - ${laptop.get('price', 'Unknown')}")

            
            # Store sentiment analysis results separately if available
            # The sentiment analysis is already included in the trend_results
            if trend_results and trend_results.get('sentiment_overview'):
                updated_fields["sentiment_analysis"] = trend_results.get('sentiment_overview')
                print(f"[MASTER AGENT] Added sentiment overview to state: {trend_results.get('sentiment_overview').get('overall_sentiment', 'Unknown')}")
            elif trend_results and isinstance(trend_results, dict) and "sentiment" in trend_results:
                # If sentiment_overview not available, try to extract from trend_results directly
                updated_fields["sentiment_analysis"] = {"results": trend_results["sentiment"]}
                print(f"[MASTER AGENT] Extracted sentiment from trend results")
            else:
                print("[MASTER AGENT] No sentiment overview available from trend analysis")
            
            print("Trend Analysis and Sentiment Analysis completed successfully.")
            
            # Add explicit mention of sentiment analysis in the insights
            if trend_results and isinstance(trend_results, dict):
                trend_results["sentiment_analysis_used"] = True
                # We already set trend_results to updated_fields["trend_insights"] above
                # This ensures any modifications to trend_results are reflected
        
    except Exception as e:
        print(f"Error executing task {current_task}: {e}")
        # Simple error state
        if current_task == "trend_and_sentiment_analysis":
            updated_fields["trend_insights"] = {"analysis": "Error in trend and sentiment analysis."}
            updated_fields["sentiment_analysis"] = {"analysis": "Error in sentiment analysis."}

    # Update task queue - use the task_queue variable we already extracted at the beginning of the function
    if task_queue:
        task_queue.pop(0)
        updated_fields["task_queue"] = task_queue
        updated_fields["current_task"] = task_queue[0] if task_queue else ""
    
    return updated_fields

def synthesize_response(state) -> dict:
    """
    Synthesize a final response based on all collected information.
    
    Args:
        state: Current agent state (can be dictionary or AgentState)
        
    Returns:
        Dictionary with updated fields
    """
    
    # Initialize the dictionary for updated fields
    updated_fields = {}
    
    # Handle both dictionary and AgentState objects
    if hasattr(state, 'get'):
        # It's a dictionary
        trend_insights = state.get("trend_insights", {})
        sentiment_analysis = state.get("sentiment_analysis", {})
        user_query = state.get("user_query", "")
        conversation_history = state.get("conversation_history", [])
    else:
        # It's a Pydantic model (AgentState)
        trend_insights = state.trend_insights or {}
        sentiment_analysis = state.sentiment_analysis or {}
        user_query = state.user_query
        conversation_history = state.conversation_history
    
    # Combine trend insights and sentiment analysis into a single comprehensive structure
    combined_insights = {}
    
    # Add trend insights if available
    if trend_insights:
        combined_insights.update(trend_insights)
        print(f"[MASTER AGENT] Using trend insights with {len(trend_insights.get('top_laptops', []))} laptop recommendations")
    else:
        print("[MASTER AGENT] No trend insights available for response synthesis")
    
    # Add sentiment analysis if available and not already included in trend insights
    if sentiment_analysis:
        # Check if sentiment is already in combined_insights to avoid duplication
        if "sentiment" not in combined_insights:
            combined_insights["sentiment"] = sentiment_analysis
            print("[MASTER AGENT] Added separate sentiment analysis to insights")
    
    # Format the prompt with only the required parameters
    prompt = RESPONSE_SYNTHESIS_PROMPT.format(
        user_query=user_query,
        trend_insights=combined_insights
    )
    
    # Call the LLM
    response_generator = llm_client.generate(
        prompt=prompt,
        system_prompt=MASTER_AGENT_SYSTEM_PROMPT
    )
    response = collect_streaming_response(response_generator)
    
    # Extract the actual response from the LLM output
    response_text = response.get("response", "")
    
    # Use the response if it's not empty, otherwise provide a fallback
    final_response = ""
    if response_text.strip():
        final_response = response_text.strip()
    else:
        # Fallback in case the LLM returns an empty response
        print("Warning: LLM returned empty response in synthesize_response. Using fallback.")
        query_type = state.get("query_type", "")
        final_response = f"I've analyzed your request about {query_type or 'tech gear'}. "
        
        if trend_insights:
            final_response += "Based on trend and sentiment analysis, here are my insights: "
            if isinstance(trend_insights, dict):
                # Add trend insights
                if "topics" in trend_insights:
                    final_response += f"\n\n**Popular Topics:**\n"
                    for topic in trend_insights["topics"]:
                        final_response += f"- {topic}\n"
                
                # Add sentiment summary if available
                if "sentiment_summary" in trend_insights:
                    final_response += f"\n**Sentiment Analysis:**\n{trend_insights['sentiment_summary']}\n"
                
                # Add recommendations if available
                if "recommendations" in trend_insights:
                    final_response += f"\n**Recommendations:**\n{trend_insights['recommendations']}\n"
        else:
            final_response += "I don't have enough information to provide specific insights at this time. Could you provide more details about what you're looking for?"
    
    # Update conversation history
    conversation_history.append({
        "role": "user",
        "content": user_query
    })
    conversation_history.append({
        "role": "assistant",
        "content": final_response
    })
    
    # Add updated fields to the return dictionary
    updated_fields["final_response"] = final_response
    updated_fields["conversation_history"] = conversation_history
    
    # Fix the f-string syntax
    if len(user_query) > 50:
        print(f"Generated response for query: '{user_query[:50]}...'")
    else:
        print(f"Generated response for query: '{user_query}'")
    print(f"Response length: {len(final_response)} characters")
    
    # For debugging
    if len(final_response) < 100:
        print(f"Warning: Short response generated: '{final_response}'")
    elif len(final_response) > 2000:
        print(f"Warning: Very long response generated: {len(final_response)} characters")
    
    return updated_fields

def ask_for_budget(state) -> dict:
    """
    Ask the user for their budget.
    
    Args:
        state: Current agent state (as a dictionary)
        
    Returns:
        Dictionary with updated fields
    """
    return {
        "waiting_for": "budget",
        "final_response": "What's your budget for this purchase?"
    }

def ask_for_use_case(state) -> dict:
    """
    Ask the user for their intended use case.
    
    Args:
        state: Current agent state (as a dictionary)
        
    Returns:
        Dictionary with updated fields
    """
    return {
        "waiting_for": "use_case",
        "final_response": "What will you primarily be using this for? For example, gaming, content creation, work, etc."
    }

def check_information_completeness(state) -> str:
    """
    Check if we have all the information needed to proceed with the task.
    
    Args:
        state: Current agent state (can be dictionary or AgentState)
        
    Returns:
        Next node to execute
    """
    # Handle both dictionary and AgentState objects
    if hasattr(state, 'get'):
        # It's a dictionary
        is_in_scope = state.get("is_in_scope", True)
        query_type = state.get("query_type", "")
        budget = state.get("budget")
        use_case = state.get("use_case")
    else:
        # It's a Pydantic model (AgentState)
        is_in_scope = state.is_in_scope
        query_type = state.query_type
        budget = state.budget
        use_case = state.use_case
    
    # If query is out of scope, don't proceed with task decomposition
    if not is_in_scope:
        return "out_of_scope_response"
    
    if query_type in ["gaming_pc_build", "pc_build", "build_recommendation"] and not budget:
        return "ask_for_budget"
    
    # For laptop recommendations, both budget and use case are important
    elif query_type in ["laptop_recommendation", "laptop_suggestion"]:
        if not budget:
            return "ask_for_budget"
        elif not use_case:
            return "ask_for_use_case"
    
    # If we have all necessary information, proceed to task decomposition
    return "decompose_tasks"

def out_of_scope_response(state) -> dict:
    """
    Handle out-of-scope queries.
    
    Args:
        state: Current agent state (as a dictionary)
        
    Returns:
        Dictionary with updated fields
    """
    # Final response is already set in understand_query, so we don't need to update anything
    # Just return an empty dict since no changes are needed
    return {}

def should_continue_tasks(state) -> str:
    """
    Determine whether to continue executing tasks or synthesize a response.
    
    Args:
        state: Current agent state (as a dictionary)
        
    Returns:
        Next node to execute
    """
    task_queue = state.get("task_queue", [])
    if task_queue:
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

# Create the agent with thread management for multi-turn conversations
agent_graph = create_agent_graph()
agent = agent_graph.compile()

# Dictionary to store thread states
thread_states = {}

def process_query(query: str, thread_id: str = None) -> dict:
    """
    Process a user query through the EggHatch AI agent.
    
    Args:
        query: User query string
        thread_id: Optional thread ID for maintaining conversation context
        
    Returns:
        Dictionary containing the response and additional context for multi-turn conversations
    """
    # Check if we have a thread_id and if it exists in our thread_states
    if thread_id and thread_id in thread_states:
        # Get the previous state for this thread
        previous_state = thread_states[thread_id]
        
        # Initialize state with previous context but new query
        initial_input_dict = {
            "user_query": query,
            "conversation_history": previous_state.get("conversation_history", []),
            "query_type": previous_state.get("query_type", ""),
            "budget": previous_state.get("budget", None),
            "use_case": previous_state.get("use_case", None),
            "requirements": previous_state.get("requirements", {}),
            "current_task": "",
            "task_queue": [],
            "waiting_for": None,
            "is_in_scope": True,
            "trend_insights": previous_state.get("trend_insights", None),
            "sentiment_analysis": previous_state.get("sentiment_analysis", None),
            "final_response": ""
        }
        print(f"Using existing thread {thread_id} with context: query_type={initial_input_dict['query_type']}, budget={initial_input_dict['budget']}")
    else:
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
        
        # Generate a new thread_id if none was provided
        if not thread_id:
            thread_id = f"thread_{len(thread_states) + 1}"
            print(f"Created new thread with ID: {thread_id}")
    
    # Run the agent
    final_state_dict = agent.invoke(initial_input_dict)
    
    # Store the final state for this thread
    thread_states[thread_id] = final_state_dict
    
    # Create a response dictionary with the final response and additional context
    response_dict = {
        "response": final_state_dict.get("final_response", "Error: Agent did not produce a final response"),
        "trend_insights": final_state_dict.get("trend_insights", {}),
        "query_type": final_state_dict.get("query_type", ""),
        "budget": final_state_dict.get("budget", None),
        "use_case": final_state_dict.get("use_case", None),
        "thread_id": thread_id  # Include the thread_id for multi-turn conversations
    }
    
    return response_dict

# Example usage
if __name__ == "__main__":
    query = "I want to build a gaming PC for about $2000 that can run Cyberpunk 2077 with ray tracing."
    response_data = process_query(query)
    print("Response:", response_data["response"])
    
    # Print trend insights if available
    if response_data["trend_insights"] and "top_laptops" in response_data["trend_insights"]:
        print(f"\nFound {len(response_data['trend_insights']['top_laptops'])} laptop recommendations:")
        for i, laptop in enumerate(response_data["trend_insights"]["top_laptops"][:3]):
            print(f"{i+1}. {laptop.get('name', 'Unknown')} - ${laptop.get('price', 'Unknown')}")

