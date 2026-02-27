from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

def main():
    print("--- 1. ĐANG ĐỌC VÀ CHIA NHỎ TÀI LIỆU ---")
    file_path = "HD-EasyCA-doanh-nghiep.pdf" # Thay bằng tên file PDF của bạn
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    # Cắt nhỏ tài liệu. Mỗi đoạn dài khoảng 1000 ký tự, phần giao nhau là 200 ký tự (để không bị đứt đoạn ngữ cảnh)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Đã cắt tài liệu thành {len(chunks)} đoạn nhỏ.")

    print("\n--- 2. ĐANG TẠO VECTOR DATABASE VỚI CHROMADB ---")
    # Khởi tạo mô hình Embedding (biến chữ thành số)
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    
    # Lưu các đoạn văn bản vào ChromaDB
    # Chạy lần đầu sẽ mất chút thời gian để máy tính tính toán
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        collection_name="legal_contract"
    )
    print("Đã lưu xong vào CSDL Vector!")

    print("\n--- 3. BẮT ĐẦU TÌM KIẾM VÀ TRẢ LỜI ---")
    # Khởi tạo mô hình Qwen2.5:7b
    llm = OllamaLLM(model="qwen2.5:7b")

    # Giả sử chúng ta muốn hỏi về một điều khoản cụ thể
    user_question = "Chứng thư số của Bên A sẽ bị thu hồi trong trường hợp nào"
    print(f"Câu hỏi: {user_question}\n")

    # Hệ thống sẽ đi tìm 3 đoạn văn có nội dung liên quan nhất đến câu hỏi
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(user_question)
    
    # Gộp nội dung 3 đoạn tìm được lại với nhau
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Đưa ngữ cảnh và câu hỏi cho AI
    prompt = f"""
    Bạn là một chuyên gia pháp lý. Hãy trả lời câu hỏi dựa trên các trích đoạn tài liệu dưới đây. 
    Nếu thông tin không có trong tài liệu, hãy nói rõ là không có, đừng tự bịa ra.
    
    [TRÍCH ĐOẠN TÀI LIỆU]:
    {context}
    
    [CÂU HỎI CỦA NGƯỜI DÙNG]: {user_question}
    
    Câu trả lời:
    """

    print("Đang suy nghĩ...")
    response = llm.invoke(prompt)
    
    print("\n--- KẾT QUẢ TỪ AI ---")
    print(response)

if __name__ == "__main__":
    main()