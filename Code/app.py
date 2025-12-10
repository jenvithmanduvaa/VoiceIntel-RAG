import streamlit as st
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
from agent import run_agent
import tempfile

client=OpenAI(api_key = "")

st.set_page_config(page_title="AI Voice Assistant", layout="wide")
st.title("🎤 AI Business Assistant (Voice + Chat)")

# Sidebar: Instructions
with st.sidebar:
    st.header("About")
    st.write("Speak or type your query. The assistant retrieves info from your business documents using an Agentic RAG pipeline.")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input options: Voice or Text
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("🎤 Speak")
    audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key="recorder")
with col2:
    st.subheader("💬 Type")
    user_query = st.text_input("Enter your query")

query_text = None
if audio and "bytes" in audio:
    st.info("Transcribing your voice...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio["bytes"])
        tmp.flush()
        with open(tmp.name, "rb") as f:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
    query_text = transcript.text
elif user_query.strip():
    query_text = user_query

if query_text:
    st.session_state.history.append({"role": "user", "content": query_text})

    with st.spinner("Agent working on your request..."):
        result = run_agent(query_text)

    # Show answer
    st.session_state.history.append({"role": "assistant", "content": result["answer"]})

    # Dynamic UI sections
    with st.expander("🔍 Behind the Scenes", expanded=False):
        st.write("**Intent Classification:**", result["intent_info"])
        st.write("**Sub-Queries:**", result["sub_queries"])
        st.write("**Documents Used:**", result["retrieved_docs"])
        st.write("**Relevant Chunks:**")
        for chunk in result["retrieved_chunks"]:
            st.markdown(f"> {chunk}")

# Chat history display
st.write("## Conversation")
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")
