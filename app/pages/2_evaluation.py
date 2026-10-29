import streamlit as st
import pandas as pd

if not st.session_state.get("logged_in"):
    st.warning("Please log in on the main page first.")
    st.stop()

st.title("Evaluation Dashboard")
st.write("Retrieval quality measured on BEIR SciFact (300 test questions) and a custom 15-question test set.")

st.subheader("Embedding model comparison (BEIR SciFact)")
models = pd.DataFrame({
    "Model": ["MiniLM-L6-v2", "mpnet-base-v2"],
    "Recall@5": [0.753, 0.733],
    "MRR": [0.605, 0.618],
    "Embedding time (s)": [60, 531],
}).set_index("Model")
st.dataframe(models)
st.bar_chart(models[["Recall@5", "MRR"]])
st.caption("MiniLM time is an estimate (about a minute). mpnet took 531s on the same corpus.")

st.subheader("Chunking strategy comparison (custom test set)")
chunks = pd.DataFrame({
    "Chunk size": ["1000 chars", "500 chars"],
    "Recall@5": [0.733, 0.667],
}).set_index("Chunk size")
st.bar_chart(chunks)
st.caption("Based on 15 questions written from paper titles, so a small and topic-level test. "
           "Treat it as indicative, not conclusive.")