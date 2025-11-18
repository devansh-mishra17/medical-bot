import os
import PyPDF2
import re
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class MedicalBookProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def load_medical_book(self, book_path):
        """Load your medical book (PDF format)"""
        print(f"📖 Loading medical book: {book_path}")
        
        if not os.path.exists(book_path):
            print(f"❌ Book not found: {book_path}")
            return None
        
        try:
            # Extract text from PDF
            text = self._extract_text_from_pdf(book_path)
            print(f"✅ Extracted {len(text)} characters from medical book")
            
            if len(text) < 100:
                print("❌ Very little text extracted - PDF might be scanned images")
                return None
            
            # Create documents with BETTER splitting
            documents = self._better_medical_split(text)
            print(f"✅ Created {len(documents)} searchable knowledge chunks")
            
            # Show what we found
            if documents:
                print(f"📝 Sample diseases found: {[doc.page_content[:50] + '...' for doc in documents[:3]]}")
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="./medical_book_db"
            )
            
            print("🎉 Medical book successfully loaded into AI brain!")
            print(f"🔍 You can now search {len(documents)} specific medical topics!")
            return vectorstore
            
        except Exception as e:
            print(f"❌ Error loading medical book: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"📄 Processing {num_pages} pages...")
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    
                    # Show progress every 50 pages
                    if (page_num + 1) % 50 == 0 or (page_num + 1) == num_pages:
                        print(f"   📃 Processed page {page_num + 1}/{num_pages}")
            
            return text
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            return ""
    
    def _better_medical_split(self, text):
        """MUCH BETTER splitting for medical encyclopedia"""
        print("🔧 Splitting into disease-specific chunks...")
        
        documents = []
        
        # Method 1: Split by disease entries (look for ALL CAPS or bold disease names)
        disease_pattern = r'([A-Z][A-Z\s]+(?:disease|syndrome|disorder|condition|cancer|itis))'
        sections = re.split(disease_pattern, text)
        
        for i in range(1, len(sections), 2):
            if i < len(sections):
                disease_name = sections[i].strip()
                disease_content = sections[i+1] if i+1 < len(sections) else ""
                
                # Clean and chunk the content
                if disease_content and len(disease_content) > 100:
                    # Split disease content into smaller chunks if too large
                    disease_chunks = self._split_disease_content(disease_content, disease_name)
                    documents.extend(disease_chunks)
        
        # Method 2: If few diseases found, split by paragraphs
        if len(documents) < 50:
            print("   🔄 Using paragraph-based splitting as backup...")
            paragraphs = re.split(r'\n\s*\n', text)
            for para in paragraphs:
                para = para.strip()
                if len(para) > 200 and len(para) < 1500:
                    # Skip table of contents and indexes
                    if not any(keyword in para.lower() for keyword in ['contents', 'index', 'volume', 'chapter']):
                        documents.append(Document(page_content=para))
        
        return documents
    
    def _split_disease_content(self, content, disease_name):
        """Split disease content into manageable chunks"""
        chunks = []
        
        # Split by major sections within disease description
        sections = re.split(r'(\b(?:Symptoms|Causes|Treatment|Diagnosis|Prevention|Prognosis)\b)', content, flags=re.IGNORECASE)
        
        current_chunk = f"{disease_name}\n\n"
        
        for i, section in enumerate(sections):
            if i % 2 == 0:  # Content
                if section.strip():
                    current_chunk += section
            else:  # Section header
                # If current chunk is substantial, save it
                if len(current_chunk) > 300:
                    chunks.append(Document(page_content=current_chunk.strip()))
                    current_chunk = f"{disease_name} - {section}\n\n"
                else:
                    current_chunk += f"{section}\n\n"
        
        # Add the last chunk
        if len(current_chunk) > 100:
            chunks.append(Document(page_content=current_chunk.strip()))
        
        return chunks
    
    def check_existing_book(self):
        """Check if we already have a medical book loaded"""
        if os.path.exists("./medical_book_db"):
            try:
                vectorstore = Chroma(
                    persist_directory="./medical_book_db",
                    embedding_function=self.embeddings
                )
                print("✅ Found existing medical book database!")
                return vectorstore
            except Exception as e:
                print(f"❌ Error loading existing database: {e}")
                return None
        return None