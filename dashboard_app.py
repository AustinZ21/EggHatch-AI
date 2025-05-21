"""
Streamlit dashboard for EggHatch AI.

This module implements a user-friendly interface for interacting with the EggHatch AI agent.
"""

import streamlit as st
import time
from app.master_agent import process_query

# Set page configuration
st.set_page_config(
    page_title="EggHatch AI",
    page_icon="🐣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    .user-message {
        background-color: #2C3E50;
        border-left: 4px solid #FF6B6B;
    }
    .assistant-message {
        background-color: #1E2A38;
        border-left: 4px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🐣 EggHatch AI</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>Your next tech upgrade is about to hatch</h2>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div style='display: flex; align-items: center; margin-bottom: 0.5rem;'><span style='font-size: 28px; margin-right: 10px;'>🐣</span><span style='font-size: 20px; font-weight: bold;'>EggHatch AI</span></div>", unsafe_allow_html=True)
    st.markdown("## About")
    st.markdown(
        "EggHatch AI is your intelligent assistant for PC building and tech gear shopping. "
        "Ask me about PC builds, component recommendations, or gaming laptops!"
    )
    
    st.markdown("## Sample Queries")
    st.markdown(
        "- **Try:** I want to buy a gaming laptop for $2000\n"
        "- **Follow-up question:** what are the reviews for these laptops\n"

        "- **Or ask:**\n"
        "- What's the best gaming laptop under $1200?\n"
        "- Is the RTX 4070 good for 1440p gaming?"
    )
    
    st.markdown("## Settings")
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=1024, max_value=8192, value=4096, step=1024)

# Initialize session state for chat history and conversation thread ID
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Initialize conversation thread ID for maintaining context across turns
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
    
# Note: We're no longer storing recommendations separately in session state
# as they're now maintained in the LangGraph thread state

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"<div class='chat-message user-message'><b>You:</b> {content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message assistant-message'><b>EggHatch AI:</b> {content}</div>", unsafe_allow_html=True)

# Chat input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Ask about PC builds, components, or tech gear:", height=100)
    submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Create a placeholder for the streaming response
        message_placeholder = st.empty()
        full_response = ""
        
        # Add an initial empty message from the assistant
        st.session_state.messages.append({"role": "assistant", "content": ""})
        
        # Get response from agent
        with st.spinner("EggHatch AI is thinking..."):
            # Get the response and any additional data, using thread_id for context
            response_data = process_query(user_input, thread_id=st.session_state.thread_id)
            
            # Extract the response text
            if isinstance(response_data, dict) and 'response' in response_data:
                response = response_data['response']
                
                # Store the thread_id for future turns
                if 'thread_id' in response_data:
                    st.session_state.thread_id = response_data['thread_id']
                    print(f"Using conversation thread: {st.session_state.thread_id}")
                
                # Log laptop recommendations if available (but don't store redundantly)
                if 'trend_insights' in response_data and 'top_laptops' in response_data['trend_insights']:
                    print(f"Found {len(response_data['trend_insights']['top_laptops'])} laptop recommendations in thread state")
            else:
                response = str(response_data)
            
            # Simulate streaming by displaying the response character by character
            for i in range(len(response)):
                full_response += response[i]
                # Update the message placeholder with the current response
                message_placeholder.markdown(f"<div class='chat-message assistant-message'><b>EggHatch AI:</b> {full_response}▌</div>", unsafe_allow_html=True)
                # Add a small delay to create a typing effect
                time.sleep(0.01)
            
            # Display the final response without the cursor
            message_placeholder.markdown(f"<div class='chat-message assistant-message'><b>EggHatch AI:</b> {full_response}</div>", unsafe_allow_html=True)
        
        # Update the last message in the chat history with the full response
        st.session_state.messages[-1]["content"] = full_response
        
        # Rerun to update the UI
        st.rerun()

# Footer
st.markdown("---")
st.markdown("EggHatch AI - Powered by Gemma 3 12B and LangGraph")
