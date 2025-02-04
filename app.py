import streamlit as st
import requests
import json
from utils.retriever_pipeline import retrieve_documents
from utils.doc_handler import process_documents
from sentence_transformers import CrossEncoder
import torch

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL="deepseek-r1:7b"                                                      #Make sure you have it installed in ollama
EMBEDDINGS_MODEL = "nomic-embed-text:latest"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"

reranker = None                                                        # 🚀 Initialize Cross-Encoder (Reranker) at the global level 
try:
    reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
except Exception as e:
    st.error(f"Failed to load CrossEncoder model: {str(e)}")


st.set_page_config(
    page_title="DeepGraph RAG-Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/DeepSeek-RAG-Chatbot',
        'Report a bug': "https://github.com/yourusername/DeepSeek-RAG-Chatbot/issues",
        'About': "# DeepGraph RAG-Pro\nAdvanced RAG System with GraphRAG, Hybrid Retrieval, and Neural Reranking"
    }
)

# Custom CSS for better dark mode support
st.markdown("""
    <style>
        /* Main app background and text */
        .stApp {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #00FF99 !important;
            text-align: center;
        }
        
        /* Chat messages */
        .stChatMessage {
            border-radius: 15px;
            padding: 15px;
            margin: 15px 0;
            border: 1px solid #333;
        }
        .stChatMessage.user {
            background-color: #2C3E50 !important;
            color: #E0E0E0 !important;
        }
        .stChatMessage.assistant {
            background-color: #1E3D59 !important;
            color: #E0E0E0 !important;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #00AAFF !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            border: none !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }
        .stButton>button:hover {
            background-color: #0088CC !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #252526 !important;
        }
        
        /* Input fields */
        .stTextInput>div>div>input {
            background-color: #2D2D2D !important;
            color: #E0E0E0 !important;
            border: 1px solid #404040 !important;
        }
        
        /* File uploader */
        .stUploadedFile {
            background-color: #2D2D2D !important;
            color: #E0E0E0 !important;
            border: 1px solid #404040 !important;
            border-radius: 8px !important;
        }
        
        /* Checkboxes and radio buttons */
        .stCheckbox, .stRadio {
            color: #E0E0E0 !important;
        }
        
        /* Sliders */
        .stSlider {
            color: #E0E0E0 !important;
        }
        .stSlider > div > div > div {
            background-color: #00AAFF !important;
        }
        
        /* Progress bars */
        .stProgress > div > div > div {
            background-color: #00AAFF !important;
        }
        
        /* Code blocks */
        code {
            background-color: #2D2D2D !important;
            color: #E0E0E0 !important;
            padding: 4px 8px !important;
            border-radius: 4px !important;
        }
    </style>
""", unsafe_allow_html=True)


                                                                                    # Manage Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


with st.sidebar:                                                                        # 📁 Sidebar
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.documents_loaded:
        with st.spinner("Processing documents..."):
            process_documents(uploaded_files,reranker,EMBEDDINGS_MODEL)
            st.success("Documents processed!")
    
    st.markdown("---")
    st.header("⚙️ RAG Settings")
    
    st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
    st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=True)
    st.session_state.enable_reranking = st.checkbox("Enable Neural Reranking", value=True)
    st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=True)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 5, 3)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 💬 Chat Interface
st.title("🤖 DeepGraph RAG-Pro")
st.caption("Advanced RAG System with GraphRAG, Hybrid Retrieval, Neural Reranking and Chat History")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def format_response(response_text, rag_info=None):
    """Format the response with RAG information and better structure."""
    # Extract thinking part if present
    thinking = ""
    final_response = response_text
    if "<think>" in response_text and "</think>" in response_text:
        parts = response_text.split("</think>")
        thinking = parts[0].replace("<think>", "").strip()
        final_response = parts[1].strip()
    
    # Format RAG information
    rag_section = ""
    if rag_info:
        matched_nodes = rag_info.get('matched_nodes', [])
        related_nodes = rag_info.get('related_nodes', [])
        doc_count = len(rag_info.get('documents', []))
        
        rag_section = f"""
🔍 **RAG Context**
• Matched Nodes: {', '.join(matched_nodes) if matched_nodes else 'None'}
• Related Nodes: {', '.join(related_nodes) if related_nodes else 'None'}
• Retrieved Documents: {doc_count}
---
"""
    
    # Format thinking section if present
    thinking_section = ""
    if thinking:
        thinking_section = f"""
💭 **Analysis Process**:
{thinking}
---
"""
    
    # Format final response with markdown
    formatted_response = f"""{rag_section}{thinking_section}
📋 **Summary**:
{final_response}
"""
    return formatted_response

SYSTEM_PROMPT = "Use the chat history to maintain context:\nChat History:\n{}\n\nAnalyze the question and context through these steps:\n1. Identify key entities and relationships\n2. Check for contradictions between sources\n3. Synthesize information from multiple contexts\n4. Formulate a structured response\n\nContext:\n{}\n\nQuestion: {}\nAnswer:"

def generate_stream(messages):
    system_prompt = messages[0]["content"]
    chat_history = "\n".join([msg["content"] for msg in messages[1:]])
    prompt = f"{system_prompt}\n{chat_history}"
    response = requests.post(
        OLLAMA_API_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": st.session_state.temperature,  # Use dynamic user-selected value
                "num_ctx": 4096
            }
        },
        stream=True
    )
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode())
            token = data.get("response", "")
            yield token

if prompt := st.chat_input("Ask about your documents..."):
    chat_history = "\n".join([msg["content"] for msg in st.session_state.messages[-5:]])
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Build context from RAG
            context = ""
            rag_info = {}
            if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
                try:
                    docs = retrieve_documents(prompt, OLLAMA_API_URL, MODEL, chat_history)
                    # Flatten and deduplicate related nodes
                    all_related_nodes = []
                    for doc in docs[:2]:
                        if doc.metadata.get("related_nodes"):
                            if isinstance(doc.metadata["related_nodes"], list):
                                all_related_nodes.extend(doc.metadata["related_nodes"])
                            else:
                                all_related_nodes.append(str(doc.metadata["related_nodes"]))
                    
                    rag_info = {
                        "matched_nodes": [str(doc.metadata.get("node_type", "Unknown")) for doc in docs[:2]],
                        "related_nodes": list(set(str(node) for node in all_related_nodes)),
                        "documents": docs
                    }
                    context = "\n".join([doc.page_content for doc in docs])
                except Exception as e:
                    st.error(f"Error retrieving documents: {str(e)}")
            
            # Generate response
            messages = [{"role": "system", "content": SYSTEM_PROMPT.format(chat_history, context, prompt)}]
            if context:
                messages.append({"role": "system", "content": f"Context:\n{context}"})
            messages.append({"role": "user", "content": prompt})
            
            # Stream response
            response = ""
            for chunk in generate_stream(messages):
                if chunk:
                    response += chunk
                    # Format and display intermediate response
                    formatted_response = format_response(response, rag_info)
                    response_placeholder.markdown(formatted_response)
            
            # Save final response
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
