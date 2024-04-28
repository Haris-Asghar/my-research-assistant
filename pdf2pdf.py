import os
from pinecone import Pinecone
from langchain_community.document_loaders import PyMuPDFLoader
from sentence_transformers import SentenceTransformer

PINE_CONE_API = os.getenv('ABSTRACTS_PINECONE_API_KEY')
TOP_K = 5
MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'
BATCH_SIZE = 1

# Initialize Pinecone
pc = Pinecone(api_key=PINE_CONE_API)
index = pc.Index('papers-index')

def generate_embeddings(texts, model_name = MODEL_NAME, batch_size=BATCH_SIZE):
    model = SentenceTransformer(model_name)
    model.prompt_name = "retrieval"
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

def query_pinecone(embedding, top_k=TOP_K):
    results = []
    try:
        query_result = index.query(vector=embedding.tolist(), top_k=top_k, include_metadata=True)
        results.append(query_result)
    except Exception as e:
        print(f"Error querying with embedding: {e}")
    return results

def extract_text (loader):
    try:
        data = loader.load()
        text = data[0].page_content 
    except Exception as e:
        print("ERROR: Could not load PDF")
        print(e)
        return ""
    start_index = text.find('Abstract')
    end_index = text.find('Introduction')
    extracted_text = ""
    if start_index != -1 and end_index != -1:
        extracted_text = text[start_index:end_index]
    elif end_index != -1:
        extracted_text = text[:end_index]
    else:
        tokens = text.split(' ')[:300] 
        # get first 300 words by default.
        # abstract usually has a max limit of 250 words
        extracted_text = ' '.join(tokens)
    return extracted_text