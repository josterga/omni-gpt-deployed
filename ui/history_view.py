import streamlit as st
from models.data_models import RAGResponse


def render_history(responses: list[RAGResponse]):
    for resp in responses:
        st.markdown(f"**{resp.user_query.timestamp}**: {resp.user_query.text}")
        st.markdown(f"- Answer: {resp.answer}")