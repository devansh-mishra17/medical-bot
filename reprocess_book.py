from chatbot.book_processor import MedicalBookProcessor
import os
import shutil

def reprocess_medical_book():
    print("🔄 Re-processing Medical Book with Better Chunking...")
    print("=" * 60)
    
    # Remove existing database to start fresh
    if os.path.exists("./medical_book_db"):
        shutil.rmtree("./medical_book_db")
        print("🗑️  Removed old database")
    
    processor = MedicalBookProcessor()
    
    # Find the medical book
    book_file = "Medical_book.pdf"
    
    if not os.path.exists(book_file):
        print(f"❌ Book not found: {book_file}")
        return False
    
    print(f"📖 Processing: {book_file}")
    print("⏳ This may take 2-3 minutes for the entire encyclopedia...")
    print("💡 The book will be split into hundreds of searchable chunks!")
    
    # Load with improved processing
    success = processor.load_medical_book(book_file)
    
    if success:
        print("🎉 Medical book re-processed successfully!")
        print("🔍 Now you can ask specific questions about diseases!")
        print("💊 Try: 'What is diabetes?' or 'Tell me about heart disease'")
        return True
    else:
        print("❌ Failed to re-process medical book")
        return False

if __name__ == "__main__":
    reprocess_medical_book()