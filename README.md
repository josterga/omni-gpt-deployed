# README

### Updated 6/17/25

## Information
This is an experimental RAG search system that uses OpenAI to respond to natural language queries using Omni Docs, Discourse, Slack as sources. 
Embeddings for Docs and Discourse are processed separately. Slack is real-time search. 

Chat history is saved and results are used as a cache which can be overridden with a newline at the end of your query.

## Environment Setup
1. create virtual environment "slack-rag" (if you change the name be sure to update .gitignore)
2. add the following environment variables
    SLACK_API_TOKEN - Slack API Token 
    OPENAI_API_KEY - OpemAI API Key

## Run
1. streamlit run main.py