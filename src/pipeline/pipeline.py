from qdrant_client import QdrantClient
from src.vectorstore import QdrantVectorStoreDB
from src.answerquery import AnswerQuery
from src.embedding import all_minilm_l6_v2
from src.settings import settings


class QAPipeline:
    """
    A class that handles the entire QA pipeline.
    """
    def __init__(self):
        self.embeddings=all_minilm_l6_v2()
        self.qdrant_client=QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


        self.vector_store = QdrantVectorStoreDB(qdrant_client=self.qdrant_client,vector_embedding= self.embeddings)
        self.answer_query = AnswerQuery()
    
    async def upload_documents(self, documents, collection_name:str="recipe"):
        """
        Upload documents to the Qdrant vector store.
        """
        await self.vector_store.upload_documents(documents, collection_name)
    

    async def answer_query_(self, query):
        """
        Answer a query using the Groq model.
        """
        return await self.answer_query.answer_query(
            vectorembedding=self.embeddings,
            query=query,
        )
    async def search_web(self, query):
        """
        Search the web for a query.
        """
        return await self.answer_query.search_web(
            query=query
        )
