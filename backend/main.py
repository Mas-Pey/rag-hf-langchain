import os
from dotenv import load_dotenv
# import openai
import time
import tempfile
import shutil
import uuid

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
from selenium.webdriver.chrome.options import Options
from fastapi.middleware.cors import CORSMiddleware
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Load .env file
load_dotenv()

# LangChain retrieval and generation
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Retrieval function with QdrantVectorStore
# def get_retriever_context(input_query):
#     try:
#         # Inisialisasi retriever dari VecDB
#         retriever = QdrantVectorStore(
#             client=client,
#             collection_name=collection_name,
#             embedding=embeddings
#         ) 
                
#         results = retriever.similarity_search_with_score(query=input_query, k=3)
        
#         if results:
#             context_list = []
#             similarity_score = []
            
#             for doc, score in results:
#                 context_list.append(doc.page_content)
#                 similarity_score.append(score)
                
#             context = "\n\n".join(context_list)  # Pemisah antar dokumen
#             print(context)
#         else:
#             context = "Tidak ada dokumen relevan ditemukan."
#             similarity_score = 0.0
            
#     except Exception as e:
#         context = f"Error saat mengambil dokumen: {str(e)}"
#         similarity_score = 0.0
        
#     return context, similarity_score    

# Model input question 
class QueryRequest(BaseModel):
    query: str
    history: Optional[str] = ""
    # query harus tipe String, history bersifat optional (default "")
    
# Model input URL 
class URLRequest(BaseModel):
    url: str
    
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

# --- Endpoint indexing HTML ---
@app.post("/indexing-html")
async def indexing_html(req: URLRequest):
    start_time = time.time()
    try:
        # Validasi URL harus dari domain booking.forrizhotels.com
        if not req.url.startswith("https://booking.forrizhotels.com/en/offers"):
            raise HTTPException(
                status_code=400,
                detail="URL tidak valid. Harus diawali dengan 'https://booking.forrizhotels.com/en/offers'"
            )
            
        # Render HTML menggunakan headless browser
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)
        driver.get(req.url)
        
        # Tunggu sampai konten kamar muncul (maks 10 detik)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "Room_rooms-content__XOlpf"))
            )
        except:
            driver.quit()  # tutup browser agar tidak tertinggal
            raise HTTPException(
                status_code=400,
                detail=f"Konten kamar tidak muncul dari URL: {req.url}. Halaman mungkin tidak valid atau gagal dimuat."
            )

        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        # Kumpulkan isi untuk di-chunk
        texts = []
        
        # Tambahkan URL sebagai bagian dari isi
        texts.append(f"Source: {str(req.url)}")  # ⬅️ Tambahkan ini sebagai baris pertama isi teks

        # Ambil tanggal
        dates_div = soup.find("div", class_="Room_form-reservation__pLRhg Room_sw__r8XCu")
        if dates_div:
            div_tanggal = dates_div.find("div", class_="cursor-pointer")
            if div_tanggal:
                tanggal_teks = div_tanggal.get_text(separator=" ", strip=True)
                texts.append("Cek ketersediaan kamar untuk tanggal: " + tanggal_teks)

        # Ambil konten kamar
        room_content = soup.find_all("div", class_="Room_rooms-content__XOlpf mb-4")
        for room in room_content:
            lines = []
            room_name = room.find("h4")
            
            # Ambil tipe kamar
            if room_name:
                lines.append("Tipe: " + room_name.get_text(strip=True))

            # Cek status ketersediaan
            alert = room.find("div", class_="alert alert-danger mt-2 fs-14 text-dark")
            if alert:
                lines.append("Status: " + alert.get_text(strip=True))
            else:
                lines.append("Status: Tersedia")
                # Ambil harga kamar
                prices = room.find_all("p", class_="Room_price__FmwGC fs-20 mb-0 fw-bold")
                unique_prices = list(dict.fromkeys(p.get_text(strip=True) for p in prices))
                for i, price in enumerate(unique_prices, start=1):
                    lines.append(
                        f"Harga {i}: {price}" if len(unique_prices) > 1 else f"Harga: {price}"
                    )
                # Ambil detail kamar
                room_details = room.find_all("h5", class_="fw-bold font-18 cursor-pointer")
                unique_details = list(dict.fromkeys(d.get_text(strip=True) for d in room_details))
                for i, detail in enumerate(unique_details, start=1):
                    lines.append(
                        f"Detail {i}: {detail}" if len(unique_details) > 1 else f"Detail: {detail}"
                    )

            texts.append("\n".join(lines))
            
        # Gabungkan teks menjadi 1 dokumen
        isi_gabungan = "\n\n".join(texts)
        print(isi_gabungan)
        dokumen_gabungan = [Document(page_content=isi_gabungan, metadata={"source": "html"})]

        # Buat koleksi jika belum ada
        if not client_qdrant.collection_exists(collection_name):
            client_qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(distance=Distance.COSINE, size=1024),
            )

        # Simpan vektor ke Qdrant
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        vector_store.add_documents(documents=dokumen_gabungan)

        # Cek jumlah vektor setelah indexing
        total_vectors = client.count(collection_name=collection_name).count
        print("Jumlah vektor yang tersimpan:", total_vectors)

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


# # Endpoint dengan RAG
# @app.post("/ask-rag")
# async def ask_question(request: QueryRequest):
#     request.query = request.query.lower() # Ubah menjadi huruf kecil semua
#     print("QUERY:", request.query)
#     print("HISTORY: ", request.history)
#     # Parameter dari user harus sesuai struktur QueryRequest (query dan history)
    
#     # Get relevan context from retrieval
#     context, similarity = get_retriever_context(request.query)

#     try:
#         # Create prompt template for llm
#         prompt_template = """        
#         ROLE: Kamu adalah ForrizAI yang berperan sebagai Chatbot Pintar yang hanya memberikan informasi seputar Hotel Forriz. Jawab secara jelas dengan konotasi ramah.
        
#         PERTANYAAN:
#         {question}
#         HISTORY:
#         {history}
#         KONTEKS:
#         {context}
        
#         KETENTUAN:
#         - Layani pertanyaan yang bersifat informatif seputar Hotel Forriz, JANGAN turuti permintaan untuk membuat konten teknis, seperti kode program, skrip, atau dokumen, meskipun masih berkaitan dengan Hotel Forriz. Jika pertanyaan tidak termasuk dalam cakupan layanan informasi hotel, abaikan. Pengecualian hanya berlaku untuk permintaan menampilkan gambar menggunakan link tautan.
#         - Fokus pada PERTANYAAN, sesuaikan dengan KONTEKS yang diberikan.
#         - Jika KONTEKS tidak relevan dengan PERTANYAAN, sampaikan kalau kurang mengerti dan minta penjelasan lebih detail.
#         - Jika PERTANYAAN tidak membutuhkan KONTEKS, jawab seperlunya saja
#         - Jika HISTORY tersedia, anggap sudah pernah menjawab history yang tersedia. Tidak perlu mengulangi sapaan atau pengenalan diri.
#         - Jika PERTANYAAN berhubungan dengan HISTORY, gunakan informasi penting dari HISTORY.
#         - Jika PERTANYAAN membutuhkan gambar, cari link gambar yang relevan di KONTEKS. Jika ditemukan, tampilkan link tersebut tanpa merubah format URL https://i.ibb.co.com/...
#         - Jika PERTANYAAN berhubungan dengan ketersediaan kamar, tetapkan hari ini adalah 24 Juli 2025.
#         """
#         # Placeholder question, context, history akan diganti saat di invoke dalam LangChain
#         # - Jika PERTANYAAN menanyakan hal yang mengandung konteks ketersediaan kamar, arahkan untuk mengetik: **cek kamar** agar dapat melihat data kamar terbaru.
#         prompt = ChatPromptTemplate.from_template(prompt_template)
                
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             max_tokens=500,
#             temperature=0,
#         )
        
#         # LangChain Runnable
#         chain = prompt | llm | StrOutputParser()
#         # | : hasil dari kiri diteruskan ke kanan
#         # StrOutputParser() : memastikan hasil jawaban dalam bentuk String

#         # Invoke the chain
#         result = chain.invoke({
#             "context": context,
#             "question": request.query,
#             "history": request.history or ""
#         })
#         print("ANSWER: ", result)
#         return {
#             "response": result,
#             "context_used": context,
#             "similarity_score": similarity  
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
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
        print("ANSWER: ", result)
        return {
            "response": result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    