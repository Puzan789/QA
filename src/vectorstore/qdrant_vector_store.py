from src.vectorstore import upload_document_existing_collection,answer_query_from_existing_collection,upload_document_new_collection
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import logging,clean_text
logger = logging.getLogger(__name__)




class QdrantVectorStoreDB:
    """
    A class that handles the Qdrant vector store database operations.
    """
    def __init__(self,qdrant_client,vector_embedding):
        self.qdrant_client = qdrant_client
        self.vector_embedding = vector_embedding

    async def create_collection(self, collection_name:str):
        """
        Create a new collection in Qdrant.
        """
        try:
            await upload_document_new_collection(self.vector_embedding, collection_name)
            logger.info(f"Collection {collection_name} created successfully.") 
            return 
        except Exception as e:
            logger.error(f"Error creating chatbot: {e}")
            raise e
  
         
    async def upload_documents(self,documents,collection_name_:str="recipe"):
        """Upload Documents to qdrant vectorstore"""

        try:
            splitter=RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            add_start_index=True
            )
            for doc in documents:
                doc.page_content = clean_text(doc.page_content)
            
            chunks=splitter.split_documents(documents)
            await upload_document_existing_collection(
                documents_=chunks,
                vector_embeddings=self.vector_embedding,
                collection_name_=collection_name_
            )
            logger.info(f"Documents uploaded successfully to {collection_name_} collection.")
            return 


        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            raise e
    

    

