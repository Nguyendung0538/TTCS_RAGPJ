import re
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

# --- CẤU HÌNH MÔ HÌNH ---
EMBEDDING_MODEL = "qwen3-embedding:8b"
LLM_MODEL = "qwen3:8b"

def load_document(file_path):
    """Đọc file theo định dạng PDF hoặc DOCX"""
    if file_path.endswith(".pdf"):
        loader = PDFPlumberLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Chỉ hỗ trợ PDF và DOCX")
    return loader.load()

def legal_text_splitter(documents, source_version):
    """
    Cắt 2 lớp (Two-tier Chunking): 
    Lớp 1: Cắt theo 'Điều' bằng Regex.
    Lớp 2: Nếu 'Điều' nào dài > 2000 ký tự, dùng RecursiveCharacterTextSplitter cắt tiếp.
    """
    final_chunks = []
    
    # Khởi tạo công cụ cắt lớp 2 (chỉ dùng khi thực sự cần)
    secondary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    for doc in documents:
        text = doc.page_content
        # Tách dựa trên chữ "Điều X"
        dieu_khoan_list = re.split(r'(?=\bĐiều\s+\d+)', text)
        
        for chunk_text in dieu_khoan_list:
            chunk_text = chunk_text.strip()
            
            # Bỏ qua các đoạn quá ngắn (rác định dạng hoặc tiêu đề)
            if len(chunk_text) > 50: 
                
                # --- KIỂM TRA ĐỘ DÀI LỚP 2 ---
                if len(chunk_text) > 2000:
                    # Chuyển text thành Document tạm thời để cắt nhỏ
                    temp_doc = Document(
                        page_content=chunk_text, 
                        metadata={"version": source_version}
                    )
                    # Cắt tiếp và đưa vào danh sách tổng
                    sub_chunks = secondary_splitter.split_documents([temp_doc])
                    final_chunks.extend(sub_chunks)
                else:
                    # Nếu độ dài an toàn (< 2000), giữ nguyên toàn bộ "Điều"
                    new_doc = Document(
                        page_content=chunk_text,
                        metadata={"version": source_version}
                    )
                    final_chunks.append(new_doc)
                    
    return final_chunks

def process_and_index_documents(path_cu, path_moi):
    """Hàm chạy luồng Ingestion -> Chunking -> Embedding"""
    docs_cu = load_document(path_cu)
    docs_moi = load_document(path_moi)

    # Cắt 2 lớp và dán nhãn
    chunks_cu = legal_text_splitter(docs_cu, "BẢN CŨ")
    chunks_moi = legal_text_splitter(docs_moi, "BẢN MỚI")
    all_chunks = chunks_cu + chunks_moi

    # Lưu vào ChromaDB
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma.from_documents(
        documents=all_chunks, 
        embedding=embeddings, 
        collection_name="contract_db"
    )
    
    return vector_db, len(all_chunks)

def compare_legal_terms(vector_db, user_question):
    """Hàm chạy luồng Retrieval -> LLM Generation"""
    # Lấy 6 đoạn liên quan nhất
    retriever = vector_db.as_retriever(search_kwargs={"k": 6})
    relevant_docs = retriever.invoke(user_question)
    
    context = ""
    for doc in relevant_docs:
        context += f"[{doc.metadata['version']}]\nNội dung: {doc.page_content}\n\n"

    # --- PROMPT: Chống trả lời máy móc và gộp trích dẫn ---
    prompt = f"""
    Bạn là một chuyên gia pháp lý. Dựa vào các trích đoạn tài liệu dưới đây, hãy tìm sự khác biệt liên quan đến yêu cầu: "{user_question}".
    
    [TRÍCH ĐOẠN TÀI LIỆU]:
    {context}
    
    [HƯỚNG DẪN TRẢ LỜI]:
    - Trình bày trực tiếp, ngắn gọn các ĐIỂM THAY ĐỔI (đã thêm gì, xóa gì, sửa gì) giữa BẢN CŨ và BẢN MỚI.
    - TUYỆT ĐỐI KHÔNG trích dẫn lại nguyên văn nội dung của hai bản (người dùng sẽ tự xem ở phần tài liệu tham khảo).
    - KHÔNG tự động thêm các từ như "Lưu ý", "Kết luận" hoặc bình luận cá nhân. Dừng sinh văn bản ngay khi tóm tắt xong.
    - Nếu trích đoạn không chứa thông tin để trả lời, chỉ được in ra đúng một câu: "Không tìm thấy thông tin thay đổi."
    
    Câu trả lời:
    """
    
    llm = OllamaLLM(model=LLM_MODEL)
    response = llm.invoke(prompt)
    
    return response, relevant_docs