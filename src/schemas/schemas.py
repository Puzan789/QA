from pydantic import BaseModel
class RagResponse(BaseModel):
    answer:str
    web_search:bool=False