import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

print("🔍 Checking Medical Book Database...")
print("=" * 50)

# Check if database exists
if os.path.exists("./medical_book_db"):
    print("✅ medical_book_db folder exists!")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = Chroma(
            persist_directory="./medical_book_db",
            embedding_function=embeddings
        )
        
        # Check how many documents are in the database
        collection = vectorstore._collection
        count = collection.count()
        print(f"✅ Database loaded successfully!")
        print(f"📊 Documents in database: {count}")
        
        # Try a test search
        test_docs = vectorstore.similarity_search("medical topics", k=2)
        print(f"🔍 Test search found {len(test_docs)} documents")
        
        if test_docs:
            print(f"📝 Sample content: {test_docs[0].page_content[:200]}...")
        else:
            print("❌ No documents found in search")
            
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print("❌ medical_book_db folder not found!")
    print("💡 The book may not have been loaded properly")