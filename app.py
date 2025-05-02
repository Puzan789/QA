import streamlit as st
from src.pipeline import QAPipeline
from langchain_community.document_loaders import CSVLoader
import os
import tempfile
import asyncio
import nest_asyncio

nest_asyncio.apply()

os.environ["STREAMLIT_WATCHER_IGNORE_PATTERNS"] = "*/torch/*"

pipeline = QAPipeline()

st.title("Recipe Q&A")

# File upload section
st.header("Upload CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    if st.button("Add Documents"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        loader = CSVLoader(file_path=tmp_path)
        data = loader.load()
        with st.spinner("Uploading documents..."):
            asyncio.run(pipeline.upload_documents(data))
        os.remove(tmp_path)
        st.success("Documents uploaded successfully.")

# Query section
st.header("Ask a Question")
query = st.text_input("Enter your question:")

if "response" not in st.session_state:
    st.session_state.response = None

# Get answer
if st.button("Get Answer") and query:
    with st.spinner("Getting answer..."):
        response = asyncio.run(pipeline.answer_query_(query))
        st.session_state.response = response  
        st.write("**Answer:**")
        st.write(response.answer)

if st.session_state.response:
    if st.session_state.response.web_search:
        if st.button("Search the web for this?"):
            with st.spinner("Searching web..."):
                web_response = asyncio.run(pipeline.search_web(query))
                if web_response:
                    st.write("**Web Search Result:**")
                    st.write(web_response)
                else:
                    st.write("No web search result found.")
