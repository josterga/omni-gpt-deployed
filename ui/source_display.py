import streamlit as st
from models.data_models import RetrievedContext


def render_sources(contexts: list[RetrievedContext]):
    with st.expander("Sources", expanded=False):
        for ctx in contexts:
            st.markdown(f"- ({ctx.source}) {ctx.text[:100]}... (score: {ctx.score:.2f})")