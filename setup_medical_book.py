from chatbot.book_processor import MedicalBookProcessor
import os

def setup_medical_book():
    print("=" * 60)
    print("📚 MEDICAL BOOK SETUP WIZARD")
    print("=" * 60)
    
    processor = MedicalBookProcessor()
    
    # Check if book already exists
    existing_db = processor.check_existing_book()
    if existing_db:
        print("✅ Medical book already loaded in the system!")
        print("   The chatbot will use your book for answers.")
        return True
    
    print("\n💡 Place your medical book PDF in this folder and name it:")
    print("   - medical_book.pdf")
    print("   - book.pdf") 
    print("   - or any PDF file")
    print("\n   The chatbot will automatically find and load it!")
    
    # Look for PDF files in current directory
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if pdf_files:
        print(f"\n📄 Found PDF files: {', '.join(pdf_files)}")
        
        # Try to load the most likely medical book
        for pdf_file in pdf_files:
            if any(keyword in pdf_file.lower() for keyword in ['medical', 'book', 'textbook', 'medicine']):
                print(f"\n🎯 Auto-selecting likely medical book: {pdf_file}")
                success = processor.load_medical_book(pdf_file)
                if success:
                    print("🎉 Medical book setup complete!")
                    print("🤖 Your chatbot is now powered by your medical book!")
                    return True
                else:
                    print("❌ Failed to load this book. Trying next...")
                    continue
        
        # If no obvious medical book, ask user
        print(f"\n🤔 Multiple PDFs found. Which one is your medical book?")
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"   {i}. {pdf_file}")
        
        try:
            choice = input(f"\nEnter number (1-{len(pdf_files)}) or press Enter to skip: ").strip()
            if choice and choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(pdf_files):
                    success = processor.load_medical_book(pdf_files[index])
                    if success:
                        print("🎉 Medical book setup complete!")
                        return True
        except:
            pass
    
    print("\n❓ No medical book found or selected.")
    book_path = input("Enter the full path to your medical book PDF (or press Enter to skip): ").strip()
    
    if book_path and os.path.exists(book_path):
        success = processor.load_medical_book(book_path)
        if success:
            print("🎉 Medical book setup complete! Your chatbot is now smarter!")
            return True
        else:
            print("❌ Failed to load the medical book.")
    else:
        print("ℹ️  No book provided. Using basic medical knowledge.")
        print("💡 You can add a book later by running this script again.")
    
    return False

if __name__ == "__main__":
    setup_medical_book()# Check what PDF files are in your main folder
