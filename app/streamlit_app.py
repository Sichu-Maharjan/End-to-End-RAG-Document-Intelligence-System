import streamlit as st
import requests
from auth import verify_user
from chat_history import save_message, get_history
import os
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

# Login gate 
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if not st.session_state.logged_in:
    st.title("🔒 Login")
    username_input = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log in"):
        if verify_user(username_input, password):
            st.session_state.logged_in = True
            st.session_state.username = username_input
            st.rerun()
        else:
            st.error("Incorrect username or password")

    st.stop()

username = st.session_state.username

# history 
if "history" not in st.session_state:
    saved = get_history(username)
    st.session_state.history = [{"question": h["question"], "answer": h["answer"]} for h in reversed(saved)]

# chat 
st.title("RAG Document Q&A")
st.write("Ask a question about the research papers in the knowledge base.")

st.subheader("Upload a new document")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Add to knowledge base"):
        with st.spinner("Processing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(
                f"{BACKEND_URL}/upload",
                params={"username": username},
                files=files,
                )
            if response.status_code != 200:
                st.error(f"Request failed ({response.status_code}): {response.text[:300]}")
                st.stop()    
            result = response.json()
        st.success(f"Added {result['filename']} ({result['chunks_added']} chunks)")

st.divider()

if "history" not in st.session_state:
    st.session_state.history = []
from chat_history import get_all_documents, delete_document_record

with st.expander("Manage Documents"):
    docs = get_all_documents()
    st.write(f"Total documents: {len(docs)}")
    for d in docs:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{d['filename']}** — {d['num_chunks']} chunks, uploaded by {d['uploaded_by']}")
        with col2:
            if st.button("Delete", key=f"del_{d['filename']}"):
                delete_document_record(d['filename'])
                st.success(f"Removed {d['filename']} from the tracking list (its chunks remain searchable).")
                st.rerun()

                
question = st.text_input("Your question:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            response = requests.post(
                f"{BACKEND_URL}/ask",
                json={"question": question, "history": st.session_state.history}
            )
            if response.status_code != 200:
                st.error(f"Request failed ({response.status_code}): {response.text[:300]}")
                st.stop()
            
            result = response.json()

            st.session_state.history.append({"question": question, "answer": result["answer"]})
            save_message(username, question, result["answer"])
        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        for source in result["sources"]:
            st.markdown(f"- {source}")

# Show full conversation history
if st.session_state.history:
    st.divider()
    st.subheader("Conversation History")
    for h in reversed(st.session_state.history):
        st.markdown(f"**Q:** {h['question']}")
        st.markdown(f"**A:** {h['answer']}")
        st.markdown("---")