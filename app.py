import streamlit as st
from src.pipeline import QAPipeline
from langchain_community.document_loaders import CSVLoader
import os
os.environ["STREAMLIT_WATCHER_IGNORE_PATTERNS"] = "*/torch/*"
import tempfile
import asyncio

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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(pipeline.upload_documents(data))
            loop.close()
        os.remove(tmp_path)
        st.success("Documents uploaded successfully.")

# Query section
st.header("Ask a Question")
query = st.text_input("Enter your question:")
if st.button("Get Answer") and query:
    with st.spinner("Getting answer..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(pipeline.answer_query_(query))
        loop.close()
    st.write("**Answer:**")
    st.write(response)