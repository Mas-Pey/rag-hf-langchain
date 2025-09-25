import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema import Document

load_dotenv()

client_llm = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

client_embed = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"], 
)

client_qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

collection_name = "koleksi"

if not client_qdrant.collection_exists(collection_name):
    client_qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,  # sesuaikan dengan embedding size (BAAI/bge-m3 = 1024)
            distance=Distance.COSINE
        ),
    )
    
collections = client_qdrant.get_collections()   
print("Collections di Qdrant Cloud:", collections)

vector_store = QdrantVectorStore(
    client=client_qdrant,
    collection_name=collection_name,
    embedding=client_embed
)

dokumen = [
    "Permainan voli dimainkan oleh dua tim yang masing-masing terdiri dari 6 pemain.",
    "Setiap tim harus melewati net tanpa menyentuh net saat memukul bola.",
]

vector_store.add_texts(dokumen)

# Cek jumlah vektor
count = client_qdrant.count(collection_name=collection_name).count
print(f"Jumlah vektor di koleksi '{collection_name}': {count}")

# Cek isi collection
points = client_qdrant.scroll(collection_name="test_collection", limit=3)
print(points)


