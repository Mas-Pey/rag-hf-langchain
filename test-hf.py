import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

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

dokumen = [
    "Permainan voli dimainkan oleh dua tim yang masing-masing terdiri dari 6 pemain.",
    "Setiap tim harus melewati net tanpa menyentuh net saat memukul bola.",
    "Satu pertandingan voli biasanya terdiri dari 5 set, dan tim yang pertama mencapai 25 poin di setiap set menang.",
    "Servis dilakukan dari belakang garis lapangan dan harus melewati net ke area lawan.",
]

vectors = []
for _, text in enumerate(dokumen):
    embedding = client_embed.feature_extraction(
        text,
        model="BAAI/bge-m3",
    )
    vectors.append(PointStruct(
        id=str(uuid.uuid4()), 
        vector=embedding, 
        payload={"content": text}
        )
    )
    
client_qdrant.upsert(
    collection_name=collection_name,
    points=vectors,
)

# Cek jumlah vektor
count = client_qdrant.count(collection_name=collection_name).count
print(f"Jumlah vektor di koleksi '{collection_name}': {count}")

# Cek isi collection
points = client_qdrant.scroll(collection_name="koleksi", limit=3)
print(points)

# completion = client_llm.chat.completions.create(
#     model="openai/gpt-oss-20b:nebius",
#     max_tokens=300,
#     messages=[
#         {
#             "role": "user",
#             "content": "Apa itu Budiman? jawab singkat saja",
#         }
#     ],
# )

# print(completion.choices[0].message)

# result = client_embed.sentence_similarity(
#     "Apa itu budiman?",
#     [
#         "Budiman adalah seorang yang bijaksana.",
#         "Budiman adalah sebuah swalayan ."
#     ],
#     model="BAAI/bge-m3",
# )

# print (result)

