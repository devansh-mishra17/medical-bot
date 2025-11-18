import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import shutil

class ChromaManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Clear existing database (for testing)
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
    
    def load_sample_medical_data(self):
        """Create some sample medical knowledge since we don't have your book yet"""
        print("📚 Creating sample medical knowledge...")
        
        sample_medical_texts = [
            "Headaches can be caused by stress, dehydration, or eye strain. Rest and hydration often help.",
            "Fever is a common symptom of infection. Normal body temperature is around 98.6°F (37°C).",
            "Common cold symptoms include runny nose, sore throat, and coughing. Rest and fluids are important.",
            "High blood pressure often has no symptoms. Regular checkups are important for detection.",
            "Diabetes management includes monitoring blood sugar, healthy eating, and regular exercise.",
            "For minor cuts, clean the wound with water and apply antibiotic ointment and a bandage.",
            "Back pain can be alleviated with proper posture, stretching, and over-the-counter pain relievers.",
            "Allergy symptoms include sneezing, itchy eyes, and runny nose. Antihistamines can provide relief.",
            "Healthy diet includes fruits, vegetables, whole grains, and lean proteins.",
            "Regular exercise for 30 minutes daily improves cardiovascular health and reduces stress."
        ]
        
        # Split the texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        from langchain.schema import Document
        documents = [Document(page_content=text) for text in sample_medical_texts]
        chunks = text_splitter.split_documents(documents)
        
        print(f"Created {len(chunks)} medical knowledge chunks")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        vectorstore.persist()
        print("✅ Medical knowledge database created!")
        return vectorstore
    
    def get_vector_store(self):
        """Get existing vector store or create new one"""
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            return self.load_sample_medical_data()