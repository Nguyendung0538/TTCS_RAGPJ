import streamlit as st
import os
import tempfile
import processing # Import toàn bộ logic từ file processing.py vừa tạo

# --- HÀM LƯU FILE TẠM ---
def save_uploaded_file(uploaded_file):
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        return None

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Trợ lý So sánh Hợp đồng", layout="wide")
st.title("⚖️ Trợ lý So sánh Văn bản Pháp lý (Local RAG)")
st.markdown("**Mô hình đang dùng:** LLM (`qwen3:8b`) | Embedding (`qwen3-embedding:8b`)")

# Khu vực Upload File
col1, col2 = st.columns(2)
with col1:
    file_cu = st.file_uploader("Tải lên BẢN CŨ (PDF/DOCX)", type=['pdf', 'docx'])
with col2:
    file_moi = st.file_uploader("Tải lên BẢN MỚI (PDF/DOCX)", type=['pdf', 'docx'])

if file_cu and file_moi:
    if st.button("🚀 Bắt đầu xử lý dữ liệu (Indexing)", type="primary"):
        with st.spinner("Hệ thống đang băm nhỏ văn bản theo 'Điều' và tạo Vector..."):
            path_cu = save_uploaded_file(file_cu)
            path_moi = save_uploaded_file(file_moi)

            # Gọi hàm xử lý từ file processing.py
            vector_db, chunk_count = processing.process_and_index_documents(path_cu, path_moi)
            
            # Lưu DB vào session để dùng cho phần hỏi đáp
            st.session_state.vector_db = vector_db
            st.success(f"✅ Đã xử lý xong {chunk_count} đoạn văn bản vào cơ sở dữ liệu!")

# Khu vực Đặt Câu Hỏi
st.divider()
st.subheader("🔍 Đặt câu hỏi đối chiếu")
user_question = st.text_input("Nhập điều/khoản bạn muốn so sánh (VD: Đối tượng áp dụng có thay đổi gì không?)")

if user_question and "vector_db" in st.session_state:
    if st.button("Phân tích & So sánh"):
        with st.spinner("AI đang tìm kiếm và đối chiếu..."):
            
            # Gọi hàm so sánh từ file processing.py
            response, citations = processing.compare_legal_terms(st.session_state.vector_db, user_question)
            
            # Hiển thị kết quả
            st.markdown("### 📝 Kết quả Đối chiếu")
            st.info(response)
            
            # Hiển thị trích dẫn (Grounding)
            with st.expander("Bấm vào đây để xem các trích đoạn gốc đã dùng làm bằng chứng (Citations)"):
                for doc in citations:
                    st.write(f"**{doc.metadata['version']}**")
                    st.write(doc.page_content)
                    st.divider()