
from fastapi import FastAPI, UploadFile,File
from src import pipeline
from src.pipeline import QAPipeline
from langchain_community.document_loaders import CSVLoader
import shutil
import os 
pipeline= QAPipeline()

app= FastAPI()

@app.post("/")
async def upload_documents(file:UploadFile=File(...)):
    """
    Create a new collection in Qdrant.
    """
    try:
        if not file.filename.endswith('.csv'):
            return {"error": "The uploaded file is not a CSV file."}
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        loader = CSVLoader(file_path=temp_path)
        data = loader.load()
        await pipeline.upload_documents(data)
        os.remove(temp_path)
        return {"message": "Documents uploaded successfully."}
        
    except Exception as e:
        raise e

@app.get("/answer")
async def answer_query(query:str):
    """
    Answer a query using the Groq model.
    """
    try:
        response = await pipeline.answer_query_(query)
        return {"response": response}
    except Exception as e:
        raise e

