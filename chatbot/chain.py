import os
import re
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Try to import Groq - it's FREE!
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
    print("✅ Groq API available - Using enhanced AI!")
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq not available - Using local mode")

class EnhancedMedicalChatbot:
    def __init__(self, book_path=None):
        self.book_path = book_path
        self.groq_available = GROQ_AVAILABLE
        self.initialize_components()
    
    def initialize_components(self):
        print("🔄 Initializing Enhanced Medical Chatbot...")
        
        # Initialize Groq if available
        if self.groq_available:
            try:
                self.llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama2-70b-4096",  # FREE and fast!
                    temperature=0.1
                )
                print("✅ Groq AI loaded successfully! 🚀")
            except Exception as e:
                print(f"❌ Groq initialization failed: {e}")
                self.groq_available = False
        
        # Load medical book
        self.vectorstore = self._load_medical_book()
        
        if self.vectorstore:
            print("✅ Medical book loaded successfully!")
        else:
            self._load_basic_knowledge()
            print("✅ Using basic medical knowledge")
    
    def _load_medical_book(self):
        """Load medical book database"""
        try:
            if not os.path.exists("./medical_book_db"):
                return None
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            vectorstore = Chroma(
                persist_directory="./medical_book_db",
                embedding_function=self.embeddings
            )
            
            # Test search
            test_results = vectorstore.similarity_search("diabetes", k=1)
            if test_results:
                print(f"✅ Medical book loaded with content")
            
            return vectorstore
                
        except Exception as e:
            print(f"❌ Error loading medical book: {e}")
            return None
    
    def _load_basic_knowledge(self):
        """Basic fallback knowledge"""
        basic_medical = [
            "Diabetes: Condition affecting blood sugar regulation. Type 1 requires insulin, Type 2 can be managed with lifestyle.",
            "Heart disease: Conditions affecting heart and blood vessels. Includes coronary artery disease and heart failure.",
            "Asthma: Chronic lung condition causing breathing difficulties. Managed with inhalers and avoiding triggers.",
            "Cancer: Abnormal cell growth that can spread. Treatments include surgery, chemotherapy, and radiation.",
        ]
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            documents = [Document(page_content=text) for text in basic_medical]
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
        except Exception as e:
            print(f"❌ Vector store failed: {e}")
            self.vectorstore = None
    
    def get_response(self, user_message):
        user_lower = user_message.lower()
        
        # Emergency check first
        emergency_response = self._check_medical_emergency(user_lower)
        if emergency_response:
            return emergency_response
        
        try:
            # Get relevant medical information
            medical_context = self._get_medical_context(user_message)
            
            if self.groq_available and medical_context:
                # Use Groq for enhanced responses
                response = self._get_groq_enhanced_response(user_message, medical_context)
            else:
                # Use basic response
                response = self._get_basic_response(medical_context, user_message)
            
            return response + "\n\n⚠️ Consult healthcare professionals for medical advice."
            
        except Exception as e:
            print(f"Error: {e}")
            return "I can provide health information. Please consult a doctor for medical advice."
    
    def _get_medical_context(self, query):
        """Get relevant medical information from the book"""
        if not self.vectorstore:
            return None
        
        try:
            docs = self.vectorstore.similarity_search(query, k=3)
            if docs:
                # Combine the most relevant information
                context = "\n\n".join([doc.page_content for doc in docs])
                return context[:2000]  # Limit context length
        except Exception as e:
            print(f"Search error: {e}")
        
        return None
    
    def _get_groq_enhanced_response(self, question, medical_context):
        """Get enhanced response using Groq AI"""
        try:
            prompt = f"""You are a medical assistant. Use the provided medical information to answer the question clearly and concisely.

MEDICAL INFORMATION:
{medical_context}

QUESTION: {question}

Please provide:
1. A clear, direct answer to the question
2. Key points in simple language
3. Important safety information
4. Reference that this is from a medical encyclopedia

Keep the response under 300 words and very easy to understand."""

            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            print(f"Groq error: {e}")
            # Fallback to basic response
            return self._get_basic_response(medical_context, question)
    
    def _get_basic_response(self, medical_context, question):
        """Basic response without Groq"""
        if medical_context:
            # Extract most relevant part
            sentences = re.split(r'[.!?]+', medical_context)
            for sentence in sentences:
                if len(sentence) > 50:
                    return sentence[:400] + ("..." if len(sentence) > 400 else "")
        
        # Fallback responses
        question_lower = question.lower()
        if 'diabetes' in question_lower:
            return "Diabetes affects blood sugar regulation. Type 1 requires insulin; Type 2 can be managed with lifestyle changes."
        elif 'heart' in question_lower:
            return "Heart disease includes conditions like coronary artery disease. Prevention involves healthy lifestyle choices."
        elif 'cancer' in question_lower:
            return "Cancer involves abnormal cell growth. Treatments include surgery, chemotherapy, and radiation therapy."
        else:
            return "I can provide information about medical conditions from authoritative sources."
    
    def _check_medical_emergency(self, query):
        """Emergency detection"""
        emergencies = {
            'chest pain': "🚨 CHEST PAIN: Could indicate heart attack. Call emergency services!",
            'heart attack': "🚨 HEART ATTACK: Call emergency services immediately!",
            'stroke': "🚨 STROKE: Call emergency services! Look for face drooping, arm weakness.",
            'difficulty breathing': "🚨 BREATHING DIFFICULTY: Emergency! Call for help!",
            'suicide': "🚨 Call emergency services or crisis helpline immediately!",
        }
        
        for condition, response in emergencies.items():
            if condition in query:
                return response + "\n\n📞 Call emergency services!"
        
        return None

print("✅ EnhancedMedicalChatbot ready with Groq support!")