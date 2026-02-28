import os
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

# --- HÀM HỖ TRỢ: ĐỌC TÀI LIỆU THEO ĐỊNH DẠNG ---
def load_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        
    if file_path.endswith(".pdf"):
        loader = PDFPlumberLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Định dạng không được hỗ trợ: {file_path}. Chỉ dùng PDF hoặc DOCX.")
    
    return loader.load()

def main():
    print("--- 1. ĐANG ĐỌC VÀ CHUẨN BỊ 2 PHIÊN BẢN TÀI LIỆU ---")
    
    # BẠN HÃY THAY TÊN FILE CỦA BẠN VÀO ĐÂY NHÉ
    file_cu = "59.2014.QH13.docx" 
    file_moi = "26.2023.QH15.docx"  

    # Đọc dữ liệu từ 2 file
    docs_cu = load_document(file_cu)
    docs_moi = load_document(file_moi)

    # Khởi tạo công cụ cắt văn bản
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    chunks_cu = text_splitter.split_documents(docs_cu)
    chunks_moi = text_splitter.split_documents(docs_moi)

    # *** GẮN NHÃN METADATA ĐỂ PHÂN BIỆT ***
    for chunk in chunks_cu:
        chunk.metadata["version"] = "BẢN CŨ"
    
    for chunk in chunks_moi:
        chunk.metadata["version"] = "BẢN MỚI"

    all_chunks = chunks_cu + chunks_moi
    print(f"Đã cắt tổng cộng {len(all_chunks)} đoạn từ 2 tài liệu.")

    print("\n--- 2. ĐANG TẠO VECTOR DATABASE---")
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
    
    # Khởi tạo CSDL thẳng trên RAM (không dùng persist_directory)
    vector_db = Chroma.from_documents(
        documents=all_chunks, 
        embedding=embeddings, 
        collection_name="contract_comparison"
    )
    print("Đã lưu xong vào ChromaDB")

    print("\n--- 3. BẮT ĐẦU TRUY XUẤT VÀ SO SÁNH ---")
    llm = OllamaLLM(model="qwen3:8b")
    
    # Câu hỏi thử nghiệm
    user_question = "Dựa vào các tài liệu được cung cấp, hãy lập bảng so sánh các trường thông tin được in trên bề mặt Thẻ Căn cước công dân (theo Luật 2014) và Thẻ Căn cước (theo Luật 2023). Mục 'Quê quán' và 'Nơi thường trú' đã được thay đổi thành gì?"
    print(f"Câu hỏi: {user_question}\n")

    # Lấy 6 đoạn liên quan nhất để đảm bảo bốc trúng cả bản cũ và mới
    retriever = vector_db.as_retriever(search_kwargs={"k": 6})
    relevant_docs = retriever.invoke(user_question)

    # Ghép nội dung tìm được cùng Nhãn (BẢN CŨ/BẢN MỚI)
    context = ""
    for doc in relevant_docs:
        context += f"[{doc.metadata['version']}]\nNội dung: {doc.page_content}\n\n"

    # Prompt đối chiếu
    prompt = f"""
    Bạn là một trợ lý pháp lý chuyên nghiệp. Nhiệm vụ của bạn là so sánh và tìm ra sự khác biệt giữa hai phiên bản của pháp lý dựa trên các trích đoạn dưới đây.
    
    [TRÍCH ĐOẠN TÀI LIỆU TÌM ĐƯỢC]:
    {context}
    
    [YÊU CẦU]: 
    1. Trả lời trực tiếp câu hỏi: {user_question}
    2. Hãy trích dẫn tóm tắt nội dung ở [BẢN CŨ] là gì và [BẢN MỚI] là gì.
    3. Chỉ ra rõ ràng điểm thay đổi (thêm, bớt, hay sửa đổi chữ nào).
    Nếu thông tin không có trong trích đoạn, hãy trả lời là "Không tìm thấy thông tin thay đổi", tuyệt đối không tự bịa ra.
    Trả lời câu hỏi bằng tiếng Việt.
    Câu trả lời của bạn:
    """

    print("AI đang phân tích và so sánh...\n")
    response = llm.invoke(prompt)
    
    print("--- KẾT QUẢ ĐỐI CHIẾU ---")
    print(response)

if __name__ == "__main__":
    main()