from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from src.settings import settings
from src.vectorstore import answer_query_from_existing_collection
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from src.schemas import RagResponse

class AnswerQuery:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        """
        Class to handle the Groq model for answering queries.
        """
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.3,
            max_tokens=512,
            api_key=settings.GROQ_API_KEY,
        )
        self.serper = GoogleSerperAPIWrapper(serper_api_key=settings.SERPER_API_KEY)

    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)

    async def answer_query(
        self, vectorembedding, query: str, collection_name: str = "recipe"
    ):
        """
        Answer a query using the Groq model.
        """
        vector_store = await answer_query_from_existing_collection(
            vectorembedding=vectorembedding,
            collection_name_=collection_name,
        )

        # Retriever
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "lambda_mult": 0.5},
        )
        template = """
        Answer using ONLY the context below:
        Context: {context}
        Question: {question}
        If context doesn't match with the question, say,I couldn’t find information about this,and set web_search to true.
        Otherwise, set web_search to false and answer only according to the context.
    

        """
        prompt = PromptTemplate.from_template(template)
        chain = (
            {
                "context": retriever|self.format_docs,
                "question": RunnablePassthrough(),

            }
            | prompt
            | self.llm.with_structured_output(
              RagResponse,
            )
            
        )

        response = chain.invoke(query)
        return response
    async def search_web(self, query: str):
        """Search the web for a query"""
        response =  self.serper.run(query)
        template = """
        Answer using ONLY the context below:
        Context: {context}
        Question: {question}
        If context doesn't match with the question, say,I couldn’t find information about this.
        """
        prompt = PromptTemplate.from_template(template)
        chain = (
            {
                "context": lambda x :response,
                "question": RunnablePassthrough(),

            }
            | prompt
            | self.llm
            )
            

        response = chain.invoke(query)
        return response.content



if __name__ == "__main__":

    async def main():
        answer_query = AnswerQuery()
        query = "What is the capital of France?"
        response = await answer_query.answer_query(query)
        print(response)

    import asyncio

    asyncio.run(main())
