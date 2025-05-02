from langchain_huggingface import HuggingFaceEmbeddings


embedding_instance = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

def all_minilm_l6_v2():
    """
    Return the  embedding instance.
    """
    return embedding_instance