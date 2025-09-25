import streamlit as st
import requests
import re
import pandas as pd

# --- Title ---
st.set_page_config(page_title="ForrizAI", layout="centered")
# --- Forriz AI ---
col1, col2 = st.columns([1,4])

with col1:
    st.image("./frontend/images/assistant.png", width="stretch")

with col2:
    st.markdown("""
    ## Halo! Saya adalah ForrizAI
    Sebuah Layanan Chatbot berbasis AI yang dirancang untuk membantu menjawab berbagai pertanyaan seputar Hotel Forriz.
    """)

# --- Sidebar Menu ---
st.sidebar.title("ℹ️ Context Info")
sidebar_context = st.sidebar.empty()
sidebar_score = st.sidebar.empty()

# --- Admin Tools (Indexing) ---
st.sidebar.title("Indexing Data")
uploaded_file = st.sidebar.file_uploader("Upload PDF untuk di-indexing", type=["pdf"])
if st.sidebar.button("📤 Chunking-PDF") and uploaded_file:
    st.session_state.do_indexing = True
    
if st.session_state.get("do_indexing"):
    st.markdown("⏳ Proses indexing sedang berjalan...")

    try:
        # Kirim file PDF ke backend
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        res = requests.post("http://127.0.0.1:8000/indexing", files=files)

        if res.status_code == 200:
            jumlah = res.json().get("jumlah_vektor", "tidak diketahui")
            durasi = res.json().get("durasi_detik", "tidak tersedia")
            st.success(f"✅ Indexing berhasil! Total vektor: {jumlah} (dalam {durasi} detik)")
        else:
            st.error(f"❌ Gagal indexing: {res.text}")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat indexing: {e}")

    st.session_state.do_indexing = False
    
# # Input URL HTML
# html_url = st.sidebar.text_input("Masukkan URL halaman untuk di-indexing")

# # Tombol untuk indexing HTML
# if st.sidebar.button("📤 Chunking-HTML") and html_url:
    # st.session_state.do_indexing_html = True    

# # Jalankan proses indexing HTML jika tombol ditekan
# if st.session_state.get("do_indexing_html"):
#     st.markdown("⏳ Proses indexing HTML sedang berjalan...")

#     try:
#         res = requests.post("http://127.0.0.1:8000/indexing-html", json={"url": html_url})
#         if res.status_code == 200:
#             jumlah = res.json().get("jumlah_vektor", "tidak diketahui")
#             durasi = res.json().get("durasi_detik", "tidak tersedia")
#             st.success(f"✅ Indexing HTML berhasil! Total vektor: {jumlah} (dalam {durasi} detik)")
#         else:
#             st.error(f"❌ Gagal indexing HTML: {res.text}")
#     except Exception as e:
#         st.error(f"❌ Terjadi kesalahan saat indexing HTML: {e}")

#     st.session_state.do_indexing_html = False


# --- Session State for history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Display Chat History ---
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        col1, col2 = st.columns(2)
        with col2:
            with st.chat_message("user"):
                st.markdown(chat["content"])
    else: 
        with st.chat_message("assistant"):
            st.markdown(chat["content"])
            if "images" in chat:
                for url in chat["images"]:
                    st.image(url, width="stretch")
            if "table" in chat:
                st.dataframe(chat["table"], width="stretch")
                    
    # Isi dari session state chat history :
    # chat_history = [
    # {"role": "user", "content": "Apa itu Forriz?"},
    # {"role": "assistant", "content": "Forriz adalah hotel...", "images": images_urls, "table": df_kamar},
    # ...
    # ]

# --- Input User ---
if user_input := st.chat_input("Tanyakan ke ForrizAI"):
    # Menampilkan langsung pesan dari user
    col1, col2 = st.columns(2)
    with col2:
        with st.chat_message("user"):
            st.markdown(user_input)
            
    # Cek apakah user minta "cek kamar"
    if user_input.lower().strip() == "cek kamar":
        try:
            df_kamar = pd.read_excel("ketersediaan_kamar.xlsx")

            with st.chat_message("assistant"):
                st.markdown("📅 **Ketersediaan Kamar Saat Ini:**")
                st.dataframe(df_kamar, hide_index=True)

            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "Berikut adalah data ketersediaan kamar saat ini.",
                "images": [],
                "table": df_kamar
            })
            
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Gagal membaca file Excel: {e}")

        # Stop agar tidak lanjut ke API
        st.stop()

    # Lanjutkan proses ke backend (jika bukan "cek kamar")
    
    # Menyiapkan payload untuk history (khusus input user saja)
    user_messages = [chat["content"] for chat in st.session_state.chat_history if chat["role"] == "user"]
    history = "\n".join([
        f"history ke-{i+1}: {content}" for i, content in enumerate(user_messages)
    ])
    # Menyiapkan Payload
    payload = {
        "query": user_input,
        # "history": "\n".join([chat["content"] for chat in st.session_state.chat_history if chat["role"] == "user"])
        "history": history
        # chat_history = [
        # {"role": "user", "content": "Apa itu Forriz?"},
        # {"role": "assistant", "content": "Forriz adalah hotel..."},
        # {"role": "user", "content": "Alamat hotel Forriz?"},
        # {"role": "assistant", "content": "Alamat hotel di..."},
        # hasil history : ["Apa itu Forriz?", "Apa ?"]
        # ]
    }
    
    # Menyimpan input user ke history 
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Post /ask 
    with st.spinner("Mencari informasi 🔎..."):
        try:
            res = requests.post("http://127.0.0.1:8000/ask-rag", json=payload)  
            res.raise_for_status() # Error jika status bukan 200
            response = res.json()
            # print(response)
            bot_reply = response["response"] # Ambil nilai dari object 'response'
        except Exception as e:
            st.error(f"❌ Failed connect to backend: {e}") 
            bot_reply = "Yah gagal :("
            response = {
                "response": "-",
                "context_used": "-",
                "similarity_score": 0.0
            }
            
        try:
            res_no_rag = requests.post("http://127.0.0.1:8000/ask-no-rag", json=payload)  
            res_no_rag.raise_for_status() # Error jika status bukan 200
            response_no_rag = res_no_rag.json()
            # print(response)
            bot_reply_no_rag = response_no_rag["response"] # Ambil nilai dari object 'response'
        except Exception as e:
            st.error(f"❌ Failed connect to backend: {e}") 
            bot_reply_no_rag = "Yah gagal :("
            response = {
                "response": "-",
            }

    # Menampilkan balasan dari hasil post rag
    with st.chat_message("assistant"):
        st.markdown("**Jawaban dengan RAG:**")
        st.markdown(bot_reply)
        
        # Regular expression (regex) untuk mendeteksi URL gambar dari string respons bot
        # Cari URL gambar dari bot_reply
        image_urls = re.findall(r"(https?://\S+\.(?:png|jpg|jpeg|gif|webp))", bot_reply)
        # image_urls = re.findall(r"(https?://[^\s]+?\.(?:png|jpg|jpeg|gif|webp))", bot_reply)

        # Tampilkan semua gambar (jika ada)
        for url in image_urls:
            st.image(url, width="stretch")       
        
    # Simpan balasan ke chat history (teks & gambar terpisah)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply, "images":image_urls})
    
    # Menampilkan balasan dari hasil post no-rag
    with st.chat_message("assistant"):
        st.markdown("**Jawaban tanpa RAG:**")
        st.markdown(bot_reply_no_rag)
        
        # Regular expression (regex) untuk mendeteksi URL gambar dari string respons bot
        # Cari URL gambar dari bot_reply
        image_urls_no_rag = re.findall(r"(https?://\S+\.(?:png|jpg|jpeg|gif|webp))", bot_reply_no_rag)
        # image_urls = re.findall(r"(https?://[^\s]+?\.(?:png|jpg|jpeg|gif|webp))", bot_reply)

        # Tampilkan semua gambar (jika ada)
        for url in image_urls_no_rag:
            st.image(url, width="stretch")       
        
    # Simpan balasan ke chat history (teks & gambar terpisah)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply_no_rag, "images":image_urls_no_rag})
    
    # Update sidebar
    with st.sidebar.expander("📄 Lihat Context Digunakan"):
        st.markdown(f"**Context:**\n\n{response['context_used']}")

    with st.sidebar.expander("📊 Similarity Scores"):
        for i, score in enumerate(response["similarity_score"], start=1):
            st.markdown(f"**Top-{i}**: {score:.3f}")
