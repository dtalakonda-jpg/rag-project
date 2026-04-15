import streamlit as st
import os
from engine import RAGEngine



# Page Configuration
st.set_page_config(
    page_title="RAG Live Dashboard",
    page_icon="🤖",
    layout="wide",
)

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "engine" not in st.session_state:
    st.session_state.engine = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# Header
st.markdown("<h1>🤖 RAG Intellect AI</h1>", unsafe_allow_html=True)

# --- Top Section (Glassmorphic) ---
with st.container():
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔑 1. Configure System")
        api_key = st.text_input("Enter Gemini API Key", type="password", help="Get your key at aistudio.google.com")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📄 2. Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, TXT)", 
            accept_multiple_files=True,
            help="Upload the documents you want the AI to analyze."
        )
        st.markdown('</div>', unsafe_allow_html=True)

# --- Process Button ---
if uploaded_files and api_key:
    if st.button("🚀 Initialize Knowledge Engine", use_container_width=True):
        with st.spinner("Processing documents and building vector index..."):
            try:
                engine = RAGEngine(api_key)
                success = engine.process_documents(uploaded_files)
                if success:
                    st.session_state.engine = engine
                    st.success("✅ Engine ready! Start chatting below.")
                else:
                    st.error("❌ No text extracted from documents.")
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")

# --- Chat Interface ---
st.markdown("---")
st.subheader("💬 Chat with your Documents")

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.engine:
        st.warning("⚠️ Please initialize the Knowledge Engine first with your API key and documents.")
    else:
        # Add user message to state
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Thinking..."):
                answer, sources = st.session_state.engine.query(prompt)
                response_placeholder.markdown(answer)
                
                # Show Sources if available
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(sources):
                            st.info(f"Source {i+1}: {doc.page_content[:200]}...")

        # Add assistant message to state
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Footer Info
st.markdown("<br><p style='text-align: center; opacity: 0.5;'>Built with Google Gemini & Pure-Python Vector Search</p>", unsafe_allow_html=True)
