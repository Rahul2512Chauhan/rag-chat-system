from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
