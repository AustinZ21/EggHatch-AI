"""
Streamlit dashboard for EggHatch AI.

This module implements a user-friendly interface for interacting with the EggHatch AI agent.
"""

import streamlit as st
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
    }
    .user-message {
        background-color: #F8F9FA;
    }
    .assistant-message {
        background-color: #E9ECEF;
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
        "- I want to build a gaming PC for $2000\n"
        "- What's the best gaming laptop under $1500?\n"
        "- Recommend components for a streaming PC\n"
        "- Is the RTX 4070 good for 1440p gaming?"
    )
    
    st.markdown("## Settings")
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=1024, max_value=8192, value=4096, step=1024)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        
        # Get response from agent
        with st.spinner("EggHatch AI is thinking..."):
            response = process_query(user_input)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the UI
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("EggHatch AI - Powered by Gemma 3 12B and LangGraph")
