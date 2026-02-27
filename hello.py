from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaLLM

def main():
    # --- BƯỚC 1: ĐỌC FILE PDF ---
    # Thay 'contract_v1.pdf' bằng đường dẫn file của bạn
    file_path = "HD-EasyCA-doanh-nghiep.pdf" 
    print(f"--- Đang đọc file: {file_path} ---")
    
    try:
        loader = PDFPlumberLoader(file_path)
        pages = loader.load()
        
        # Lấy nội dung trang đầu tiên để thử nghiệm
        context = pages[0].page_content
        print("Đã đọc xong trang 1.")
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    # --- BƯỚC 2: KẾT NỐI VỚI QWEN 2.5 QUA OLLAMA ---
    print("--- Đang kết nối với Qwen2.5:7b ---")
    llm = OllamaLLM(model="qwen2.5:7b")

    # --- BƯỚC 3: TẠO PROMPT VÀ HỎI ĐÁP ---
    prompt = f"""
    Bạn là một trợ lý pháp lý thông minh.
    Dưới đây là nội dung của một hợp đồng.
    {context}
    
    Câu hỏi: Bên cung cấp dịch vụ là ai.
    """

    print("--- Đang chờ AI trả lời... ---")
    response = llm.invoke(prompt)

    print("\n--- KẾT QUẢ TỪ AI ---")
    print(response)

if __name__ == "__main__":
    main()