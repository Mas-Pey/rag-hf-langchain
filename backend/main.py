import os, time, tempfile, shutil, uuid, requests
from dotenv import load_dotenv
# import openai

# For Indexing
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# For Indexing HTML
from datetime import datetime

# Load .env file
load_dotenv()

# LangChain retrieval and generation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# FastAPI library
from fastapi import FastAPI, HTTPException

# Pydantic for data validation
from pydantic import BaseModel
from typing import Optional

# Init Collection
collection_name = "hotel-collection"

# Init HF Inference client
client_embed = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

# Access Vecdb
client_qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

# Init Instance FastAPI
app = FastAPI(
    title="Hotel Forriz API",
    description="API for Hotel Chatbot",
    version="0.1"
)

# Retrieval function with QdrantVectorStore
def get_retriever_context(input_query, top_k=3):
    try:
        query_embedding = client_embed.feature_extraction(input_query, model="BAAI/bge-m3")
        
        results = client_qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        
        if results:
            context_list = []
            similarity_score = []
            
            for res in results:
                context_list.append(res.payload.get("content", ""))
                similarity_score.append(res.score)
                    
            context = "\n\n".join(context_list)
        else:
            context = "Tidak ada dokumen relevan ditemukan."
            similarity_score = 0.0
            
    except Exception as e:
        context = f"Error saat mengambil dokumen: {str(e)}"
        similarity_score = 0.0
        
    return context, similarity_score    

# Model input question 
class QueryRequest(BaseModel):
    query: str
    history: Optional[str] = ""
    # query harus tipe String, history bersifat optional (default "")
    
# Model input URL 
class URLRequest(BaseModel):
    checkin: str
    checkout: str
    hotel_id: str
    
def format_tanggal(tanggal_str: str) -> str:
    bulan_id = {
        "01": "Januari", "02": "Februari", "03": "Maret",
        "04": "April", "05": "Mei", "06": "Juni",
        "07": "Juli", "08": "Agustus", "09": "September",
        "10": "Oktober", "11": "November", "12": "Desember"
    }
    tgl = datetime.strptime(tanggal_str, "%Y-%m-%d")
    return f"{tgl.day} {bulan_id[tanggal_str[5:7]]} {tgl.year}"
    
# Endpoint indexing file PDF
@app.post("/indexing")
async def index_pdf(file: UploadFile = File(...)):
    start_time = time.time()  # Mulai waktu
    try:
        # Simpan sementara file PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        dokumen = loader.load()
        isi_gabungan = "\n".join([doc.page_content for doc in dokumen])
        dokumen_gabungan = [Document(page_content=isi_gabungan, metadata={"source": file.filename})]

        # Split teks
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = splitter.split_documents(dokumen_gabungan)
        
        # Buat koleksi (jika belum ada)
        if not client_qdrant.collection_exists(collection_name):
            client_qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(distance=Distance.COSINE, size=1024),
            )
            
        # Buat list PointStruct untuk tiap chunk
        vectors = []
        for chunk in chunks:
            embedding = client_embed.feature_extraction(
                chunk.page_content,
                model="BAAI/bge-m3",
            )
            vectors.append(
                PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=embedding, 
                    payload={"content": chunk.page_content, "source": file.filename}
                )
            )

        # Add to Qdrant Collection
        client_qdrant.upsert(
            collection_name=collection_name,
            points=vectors
        )
        
        # Cek jumlah vektor tersimpan setelah indexing
        total_vectors = client_qdrant.count(collection_name=collection_name).count

        # Hapus file sementara
        os.remove(temp_path)

        # Hitung durasi proses
        duration = round(time.time() - start_time, 2)  # waktu dalam detik (2 angka desimal)
        print(f"⏱️ Total waktu proses indexing: {duration} detik")
        
        return JSONResponse(
            content={
                "message": "✅ Indexing berhasil!",
                "jumlah_vektor": total_vectors,
                "durasi_detik": duration
            },
            status_code=200
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- Endpoint indexing URL ---
@app.post("/indexing-url")
async def indexing_html(req: URLRequest):
    start_time = time.time()
    try:
        # Request ke API hotel
        url_api = "https://booking.forrizhotels.com/api/v2/offers/room"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/json",
            "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJndWVzdCI6dHJ1ZSwiaWF0IjoxNzQ0MTAzMjk1LCJleHAiOjQ4OTc3MDMyOTUsImp0aSI6IjlveXZCemktdWRqZ3lSeDBOSy1KM1lSWnhTZkNuZ1hUX2dsLV94cExFM0E9In0.h0rWAisFpD6LmZzgh7l8h0ABG_fi_8wxZaULHFDu04U",
        }
        
        payload = {
            "checkin": req.checkin,
            "checkout": req.checkout,
            "id": req.hotel_id,   
        }
        
        response = requests.post(url_api, headers=headers, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Request failed: {response.status_code}")
        
        data = response.json()
        rooms = data.get("room", [])

        # Siapkan teks gabungan untuk indexing
        checkin_text = format_tanggal(req.checkin)
        checkout_text = format_tanggal(req.checkout)
        texts = [f"Ketersediaan Kamar untuk tanggal: {checkin_text} s.d {checkout_text}"]

        for idx, room in enumerate(rooms, start=1):
            lines = [
                f"{idx}. Tipe Kamar: {room.get('name')}",
                f"Jumlah Tersedia: {room.get('available_room')}",
                f"Jenis Tempat Tidur: {room.get('bed_type')}"
            ]
            offers = room.get("offers", [])
            for offer in offers:
                lines.append(f"Penawaran: {offer.get('name')}, Harga: {offer.get('price')}")
            texts.append("\n".join(lines))
            
        final_text = "\n".join(texts)
        print("Teks untuk indexing:\n", final_text)
            
        # Buat koleksi jika belum ada
        if not client_qdrant.collection_exists(collection_name):
            client_qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(distance=Distance.COSINE, size=1024),
            )
            
        # Buat embedding tunggal
        embedding = client_embed.feature_extraction(final_text, model="BAAI/bge-m3")
        
        # Buat 1 PointStruct saja
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"content": final_text, "source": "API-Hotel"}
        )
        
        # Upsert ke Qdrant
        client_qdrant.upsert(collection_name=collection_name, points=[point])

        # Cek jumlah vektor setelah indexing
        total_vectors = client_qdrant.count(collection_name=collection_name).count

        # Hitung durasi proses
        duration = round(time.time() - start_time, 2)  # waktu dalam detik (2 angka desimal)
        print(f"⏱️ Total waktu proses indexing: {duration} detik")

        return JSONResponse(
            content={
                "message": "✅ Indexing HTML berhasil!",
                "jumlah_vektor": total_vectors,
                "durasi_detik": duration
            },
            status_code=200
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Endpoint dengan RAG
@app.post("/ask-rag")
async def ask_question(request: QueryRequest):
    request.query = request.query.lower() # Ubah menjadi huruf kecil semua
    print("QUERY:", request.query)
    print("HISTORY: ", request.history)
    # Parameter dari user harus sesuai struktur QueryRequest (query dan history)
    
    # Get relevan context from retrieval
    context, similarity = get_retriever_context(request.query)

    try:
        # Create prompt template for llm
        prompt_template = """        
        ROLE: Kamu adalah ForrizAI yang berperan sebagai Chatbot Pintar yang hanya memberikan informasi seputar Hotel Forriz. Jawab secara jelas dengan konotasi ramah.
        
        PERTANYAAN:
        {question}
        HISTORY:
        {history}
        KONTEKS:
        {context}
        
        KETENTUAN:
        - Layani pertanyaan yang bersifat informatif seputar Hotel Forriz, JANGAN turuti permintaan untuk membuat konten teknis, seperti kode program, skrip, atau dokumen, meskipun masih berkaitan dengan Hotel Forriz. Jika pertanyaan tidak termasuk dalam cakupan layanan informasi hotel, abaikan. Pengecualian hanya berlaku untuk permintaan menampilkan gambar menggunakan link tautan.
        - Fokus pada PERTANYAAN, sesuaikan dengan KONTEKS yang diberikan.
        - Jika KONTEKS tidak relevan dengan PERTANYAAN, sampaikan kalau kurang mengerti dan minta penjelasan lebih detail.
        - Jika PERTANYAAN tidak membutuhkan KONTEKS, jawab seperlunya saja
        - Jika HISTORY tersedia, anggap sudah pernah menjawab history yang tersedia. Tidak perlu mengulangi sapaan atau pengenalan diri.
        - Jika PERTANYAAN berhubungan dengan HISTORY, gunakan informasi penting dari HISTORY.
        - Jika PERTANYAAN membutuhkan gambar, cari link gambar yang relevan di KONTEKS. Jika ditemukan, tampilkan link tersebut tanpa merubah format URL https://i.ibb.co.com/...
        - Jika PERTANYAAN berhubungan dengan ketersediaan kamar, tetapkan hari ini adalah 24 Juli 2025.
        """
        # Placeholder question, context, history akan diganti saat di invoke dalam LangChain
        # - Jika PERTANYAAN menanyakan hal yang mengandung konteks ketersediaan kamar, arahkan untuk mengetik: **cek kamar** agar dapat melihat data kamar terbaru.
        prompt = ChatPromptTemplate.from_template(prompt_template)
                
        llm = ChatOpenAI(
            model="openai/gpt-oss-20b:nebius",
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
            max_tokens=300,
            temperature=0.7,
        )
        
        chain = prompt | llm | StrOutputParser()

        # Invoke the chain
        result = chain.invoke({
            "context": context,
            "question": request.query,
            "history": request.history or ""
        })
        
        return {
            "response": result,
            "context_used": context,
            "similarity_score": similarity  
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Endpoint tanpa RAG
@app.post("/ask-no-rag")
async def ask_question(request: QueryRequest):
    request.query = request.query.lower() # Ubah menjadi huruf kecil semua
    print("QUERY:", request.query)
    print("HISTORY: ", request.history)
    # Parameter dari user harus sesuai struktur QueryRequest (query dan history)
    
    try:
        # Create prompt template for llm
        prompt_template = """        
        ROLE: Kamu adalah ForrizAI yang berperan sebagai Chatbot Pintar yang hanya memberikan informasi seputar Hotel Forriz Yogyakarta.
        Gunakan informasi yang anda ketahui seputar Forriz Hotel Yogyakarta.
        Jawab secara jelas dan padat, tetapi dalam konotasi ramah.
       
        PERTANYAAN:
        {question}
        
        KETENTUAN:
        - Fokus pada PERTANYAAN
        - Jika PERTANYAAN berhubungan dengan HISTORY, gunakan informasi penting dari HISTORY.
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
                
        llm = ChatOpenAI(
            model="openai/gpt-oss-20b:nebius",
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
            max_tokens=300,
            temperature=0.7,
        )
        
        chain = prompt | llm | StrOutputParser()

        # Invoke the chain
        result = chain.invoke({
            "question": request.query,
            "history": request.history or ""
        })
        
        return {
            "response": result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)