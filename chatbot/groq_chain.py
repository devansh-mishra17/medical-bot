import os
import re
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class GroqMedicalChatbot:
    def __init__(self, book_path=None):
        self.book_path = book_path
        self.groq_available = False
        self.llm = None
        self.initialize_components()
    
    def initialize_components(self):
        print("🔄 Initializing Medical Chatbot with Groq...")
        
        # Initialize Groq if API key available
        self._initialize_groq()
        
        # Load medical book
        self.vectorstore = self._load_medical_book()
        
        if self.vectorstore:
            print("✅ Medical book loaded successfully!")
        else:
            self._load_basic_knowledge()
            print("✅ Using basic medical knowledge")
    
    def _initialize_groq(self):
        """Initialize Groq with current model names"""
        try:
            from langchain_groq import ChatGroq
            
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key or not groq_api_key.startswith('gsk_'):
                print("❌ No valid Groq API key found - using local mode")
                return
            
            # Current available models
            model_names = [
                "llama-3.1-8b-instant",      # Fast and efficient
                "llama-3.1-70b-versatile",   # Powerful and accurate
                "llama-3.2-1b-preview",      # Lightweight
                "llama-3.2-3b-preview",      # Balanced
                "llama-3.2-90b-preview",     # Most powerful
            ]
            
            for model_name in model_names:
                try:
                    print(f"🔄 Trying to load model: {model_name}")
                    self.llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name=model_name,
                        temperature=0.1,
                        max_tokens=600
                    )
                    
                    # Quick test
                    test_response = self.llm.invoke("Say 'READY' if working.")
                    if "READY" in test_response.content:
                        self.groq_available = True
                        print(f"✅ Groq AI loaded with {model_name}! 🚀")
                        break
                    else:
                        print(f"❌ {model_name} test response issue")
                        
                except Exception as e:
                    error_msg = str(e)
                    if 'decommissioned' in error_msg:
                        print(f"❌ {model_name} - DEPRECATED")
                    elif 'not exist' in error_msg:
                        print(f"❌ {model_name} - NOT FOUND")
                    else:
                        print(f"❌ {model_name} - FAILED: {error_msg[:100]}")
                    continue
                    
            if not self.groq_available:
                print("❌ No working Groq models found - using local mode")
                
        except Exception as e:
            print(f"❌ Groq initialization failed: {e}")
    
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
            return vectorstore
                
        except Exception as e:
            print(f"❌ Error loading medical book: {e}")
            return None
    
    def _load_basic_knowledge(self):
        """Basic fallback knowledge"""
        basic_medical = [
            "Diabetes management involves both medication and lifestyle changes.",
            "Heart disease prevention includes diet, exercise, and medications.",
            "Asthma treatment uses inhalers and trigger avoidance.",
        ]
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            documents = [Document(page_content=text) for text in basic_medical]
            self.vectorstore = Chroma.from_documents(documents=documents, embedding=self.embeddings)
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
            # Use Groq if available, otherwise basic search
            if self.groq_available:
                return self._get_groq_enhanced_response(user_message)
            else:
                return self._get_basic_response(user_message)
            
        except Exception as e:
            print(f"Error: {e}")
            return "I can provide health information. Please consult a doctor for medical advice.\n\n⚠️ Consult healthcare professionals."
    
    def _get_groq_enhanced_response(self, question):
        """Use Groq to generate intelligent medical responses"""
        try:
            # Get medical context from book
            medical_context = self._get_medical_context(question)
            
            prompt = f"""You are a medical expert assistant. Answer this medical question clearly and helpfully.

Question: {question}

"""
            
            if medical_context:
                prompt += f"Reference medical information:\n{medical_context[:500]}\n\n"
            
            prompt += """Please provide:
- Clear, accurate answer
- Simple explanations
- Practical information
- Important safety notes

Answer:"""

            response = self.llm.invoke(prompt)
            return response.content + "\n\n⚠️ Consult healthcare professionals for medical advice."
            
        except Exception as e:
            print(f"Groq error: {e}")
            return self._get_basic_response(question)
    
    def _get_medical_context(self, query):
        """Get relevant medical information from the book"""
        if not self.vectorstore:
            return None
        
        try:
            docs = self.vectorstore.similarity_search(query, k=2)
            if docs:
                # Clean the content
                clean_content = []
                for doc in docs:
                    text = doc.page_content
                    # Remove irrelevant content
                    lines = text.split('\n')
                    clean_lines = [line.strip() for line in lines if len(line.strip()) > 20 and 'contents' not in line.lower()]
                    clean_content.extend(clean_lines)
                
                return '\n'.join(clean_content)[:800]
        except Exception as e:
            print(f"Search error: {e}")
        
        return None
    
    def _get_basic_response(self, question):
        """Basic response without Groq"""
        question_lower = question.lower()
        
        # Enhanced basic responses
        if 'diabetes' in question_lower and 'compare' in question_lower:
            return """**Medication vs Lifestyle for Diabetes**

**Medications:**
• Insulin injections (Type 1)
• Metformin and other pills (Type 2)
• Quick blood sugar control
• Essential for some cases

**Lifestyle Changes:**
• Healthy diet low in sugar
• Regular exercise
• Weight management
• Long-term health benefits

**Best Approach:** Combination of both for comprehensive diabetes management.

⚠️ Consult healthcare professionals for medical advice."""
        
        elif 'diabetes' in question_lower:
            return "Diabetes management combines medications (insulin, metformin) with lifestyle changes (diet, exercise) for optimal blood sugar control.\n\n⚠️ Consult healthcare professionals for medical advice."
        
        elif 'heart' in question_lower:
            return "Heart disease treatment includes medications, medical procedures, and lifestyle modifications like healthy eating and regular physical activity.\n\n⚠️ Consult healthcare professionals for medical advice."
        
        elif 'cancer' in question_lower:
            return "Cancer treatment options include surgery, chemotherapy, radiation therapy, immunotherapy, and targeted therapies depending on cancer type and stage.\n\n⚠️ Consult healthcare professionals for medical advice."
        
        else:
            medical_context = self._get_medical_context(question)
            if medical_context:
                # Extract first relevant sentence
                sentences = re.split(r'[.!?]+', medical_context)
                for sentence in sentences:
                    if len(sentence) > 40:
                        return sentence[:250] + "...\n\n⚠️ Consult healthcare professionals for medical advice."
            
            return "I can provide medical information from authoritative sources. Please consult healthcare professionals for personalized medical advice.\n\n⚠️ Consult healthcare professionals for medical advice."
    
    def _check_medical_emergency(self, query):
        """Emergency detection"""
        emergencies = {
            'chest pain': "🚨 CHEST PAIN: Could indicate heart attack! Call emergency services immediately!",
            'heart attack': "🚨 HEART ATTACK: Call emergency services now! Symptoms: chest pain, shortness of breath.",
            'stroke': "🚨 STROKE: Call emergency services! Look for face drooping, arm weakness, speech difficulty.",
        }
        
        for condition, response in emergencies.items():
            if condition in query:
                return response + "\n\n📞 Call emergency services!"
        
        return None

print("✅ GroqMedicalChatbot ready!")