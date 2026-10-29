import streamlit as st
from monitoring import load_requests, count_by_level

if not st.session_state.get("logged_in"):
    st.warning("Please log in on the main page first.")
    st.stop()

st.title("System Monitoring")

df = load_requests()
errors = count_by_level("ERROR")

if df.empty:
    st.info("No requests logged yet. Ask some questions first.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total questions", len(df))
c2.metric("Avg response time", f"{df['total_s'].mean():.1f}s")
c3.metric("Errors", errors)
c4.metric("'I don't know' rate", f"{df['idk'].mean() * 100:.0f}%")

st.subheader("Where the time goes")
st.write(f"Average retrieval: **{df['retrieval_s'].mean():.2f}s**, "
         f"average LLM generation: **{df['llm_s'].mean():.2f}s**")
st.bar_chart(df[["retrieval_s", "llm_s"]])

st.subheader("Slowest questions")
st.dataframe(df.sort_values("total_s", ascending=False).head(5)[["question", "total_s", "idk"]])

st.subheader("All requests")
st.dataframe(df)