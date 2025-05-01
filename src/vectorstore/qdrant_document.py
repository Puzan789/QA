from langchain_qdrant import QdrantVectorStore,RetrievalMode
from src.settings import settings

async def answer_query_from_existing_collection(vectorembedding,collection_name_:str):
    vectorstore=QdrantVectorStore.from_existing_collection(
        embedding=vectorembedding,
        collection_name=collection_name_,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        retrieval_mode=RetrievalMode.DENSE
    )
    return vectorstore

async def upload_document_existing_collection(documents_,vector_embeddings,collection_name_):
    vector_store=QdrantVectorStore.from_documents(
        documents=documents_,
        embedding=vector_embeddings,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=True,
        collection_name=collection_name_,
        retrieval_mode=RetrievalMode.DENSE,
        timeout=None
    )

async def upload_document_new_collection(vector_embeddings,collection_name_):
    vector_store=QdrantVectorStore.from_documents(
        documents=[],
        embedding=vector_embeddings,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=True,
        collection_name=collection_name_,
        retrieval_mode=RetrievalMode.DENSE,
        force_recreate=True,
        timeout=None
    )