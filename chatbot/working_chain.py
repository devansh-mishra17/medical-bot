from langchain_core.documents import Document
import os
import re

print("🔄 Loading Medical Chatbot Components...")

# Try multiple import options for embeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    print("✅ Using langchain_community imports")
except ImportError:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        print("✅ Using langchain_huggingface imports")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        # Fallback classes
        class HuggingFaceEmbeddings:
            def __init__(self, model_name):
                self.model_name = model_name
        class Chroma:
            def __init__(self, persist_directory=None, embedding_function=None):
                pass
            @classmethod
            def from_documents(cls, documents, embedding, persist_directory):
                return cls()
            def similarity_search(self, query, k=1):
                return []

class MedicalChatbot:
    def __init__(self, book_path=None):
        self.book_path = book_path
        self.initialize_components()
    
    def initialize_components(self):
        print("🔄 Initializing Medical Chatbot...")
        
        # Try to load medical book first
        self.vectorstore = self._load_medical_book()
        
        if self.vectorstore:
            print("✅ Medical book loaded successfully! 🎉")
        else:
            # Fallback to basic medical knowledge
            self._load_basic_knowledge()
            print("✅ Using basic medical knowledge")
    
    def _load_medical_book(self):
        """Try to load medical book with better search"""
        try:
            # Check if medical book database exists
            if not os.path.exists("./medical_book_db"):
                print("❌ No medical book database found")
                return None
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            vectorstore = Chroma(
                persist_directory="./medical_book_db",
                embedding_function=self.embeddings
            )
            
            # Test if database has content
            test_results = vectorstore.similarity_search("diabetes", k=1)
            if test_results:
                print(f"✅ Medical book loaded with {len(test_results)} test results")
            else:
                print("⚠️ Medical book database may be empty")
            
            return vectorstore
                
        except Exception as e:
            print(f"❌ Error loading medical book: {e}")
            return None
    
    def _load_basic_knowledge(self):
        """Load basic medical knowledge as fallback"""
        basic_medical = [
            "Headaches can be caused by stress, dehydration, or eye strain. Rest and hydration often help. Over-the-counter pain relievers like ibuprofen can provide relief.",
            "Fever is a common symptom of infection. Normal body temperature is 98.6°F. Rest, fluids, and fever reducers like acetaminophen can help manage fever.",
            "Common cold symptoms include runny nose, sore throat, and coughing. Treatment focuses on rest, hydration, and over-the-counter cold medications.",
            "Back pain can be alleviated with proper posture, stretching exercises, and over-the-counter pain relievers. Physical therapy may help chronic cases.",
            "Allergy symptoms include sneezing, itchy eyes, and runny nose. Antihistamines can provide relief. Avoid known allergens when possible.",
            "Healthy diet includes fruits, vegetables, whole grains, and lean proteins. Limit processed foods and sugar for better health.",
            "Regular exercise for 30 minutes daily improves cardiovascular health and reduces stress. Include both cardio and strength training.",
            "Stress management techniques include deep breathing, meditation, regular exercise, and maintaining work-life balance.",
            "Good sleep hygiene includes consistent sleep schedule, dark quiet environment, and avoiding screens before bedtime.",
            "High blood pressure management includes reducing salt intake, regular exercise, and taking prescribed medications.",
            "Diabetes care involves monitoring blood sugar, eating balanced meals, regular exercise, and taking medications as prescribed.",
            "Asthma symptoms include wheezing, coughing, and shortness of breath. Inhalers and avoiding triggers can help manage asthma.",
            "Arthritis causes joint pain and stiffness. Pain relievers, exercise, and physical therapy can provide relief.",
            "Depression is a mood disorder that can be treated with therapy, medication, and lifestyle changes. Seek professional help.",
            "Anxiety disorders can be managed through therapy, relaxation techniques, and in some cases, medication prescribed by a doctor.",
            "Heart disease prevention includes healthy diet, regular exercise, not smoking, and managing blood pressure and cholesterol.",
            "Cancer screening and early detection are important. Regular checkups and following medical guidelines can save lives.",
            "Stroke symptoms include sudden numbness, confusion, trouble speaking, and vision problems. Call emergency services immediately.",
            "Heart attack symptoms include chest pain, shortness of breath, nausea, and sweating. Call emergency services immediately.",
            "Diabetes symptoms include increased thirst, frequent urination, fatigue, and blurred vision. See a doctor for diagnosis."
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
            print(f"✅ Loaded {len(basic_medical)} medical facts")
            
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
            if self.vectorstore:
                # Search medical knowledge
                docs = self.vectorstore.similarity_search(user_message, k=2)
                
                if docs:
                    response = self._format_concise_response(docs, user_message)
                else:
                    response = self._get_concise_fallback(user_lower)
            else:
                response = self._get_concise_fallback(user_lower)
            
            # Add medical disclaimer (shorter)
            return response + "\n\n⚠️ Consult healthcare professionals for medical advice."
            
        except Exception as e:
            print(f"Error in get_response: {e}")
            return "I can provide general health information. Please consult a doctor for medical advice."
    
    def _format_concise_response(self, docs, question):
        """Extract only the most relevant information and make it concise"""
        full_content = docs[0].page_content
        
        # Extract the most relevant sentence or short paragraph
        concise_response = self._extract_most_relevant_part(full_content, question)
        
        # Limit response length
        if len(concise_response) > 500:
            concise_response = concise_response[:497] + "..."
        
        return concise_response
    
    def _extract_most_relevant_part(self, content, question):
        """Extract the most relevant part of the content based on the question"""
        question_lower = question.lower()
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        # Look for sentences that directly answer the question
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Check if sentence is relevant to the question
            if self._is_relevant(sentence, question_lower):
                relevant_sentences.append(sentence)
                
                # Stop when we have enough content
                if len(' '.join(relevant_sentences)) > 200:
                    break
        
        if relevant_sentences:
            return ' '.join(relevant_sentences)
        
        # If no specific matches, return the first substantial part
        for sentence in sentences:
            if len(sentence) > 50:
                return sentence[:400] + ("..." if len(sentence) > 400 else "")
        
        # Fallback to first part of content
        return content[:400] + ("..." if len(content) > 400 else "")
    
    def _is_relevant(self, sentence, question):
        """Check if a sentence is relevant to the question"""
        sentence_lower = sentence.lower()
        
        # Question type detection
        if 'what is' in question:
            return any(word in sentence_lower for word in ['is', 'defined as', 'means', 'refers to'])
        elif 'symptoms' in question:
            return any(word in sentence_lower for word in ['symptom', 'sign', 'experience', 'feel'])
        elif 'treatment' in question or 'treat' in question:
            return any(word in sentence_lower for word in ['treat', 'therapy', 'medication', 'drug', 'cure'])
        elif 'cause' in question:
            return any(word in sentence_lower for word in ['cause', 'due to', 'because', 'result from'])
        elif 'diagnos' in question:
            return any(word in sentence_lower for word in ['diagnos', 'test', 'detect', 'identify'])
        
        # General relevance - sentence contains key terms from question
        question_words = set(question.split())
        sentence_words = set(sentence_lower.split())
        common_words = question_words.intersection(sentence_words)
        
        return len(common_words) >= 2  # At least 2 common words
    
    def _get_concise_fallback(self, query):
        """Very concise fallback responses"""
        query_lower = query.lower()
        
        if 'diabetes' in query_lower:
            return "Diabetes is a condition where the body can't properly regulate blood sugar. There are two main types: Type 1 (insulin-dependent) and Type 2 (often lifestyle-related)."
        
        elif any(word in query_lower for word in ['heart', 'cardiac']):
            return "Heart disease refers to conditions affecting the heart and blood vessels. Includes coronary artery disease, heart failure, and arrhythmias."
        
        elif 'cancer' in query_lower:
            return "Cancer is abnormal cell growth that can spread. Treatments include surgery, chemotherapy, radiation, and immunotherapy."
        
        elif 'asthma' in query_lower:
            return "Asthma is a chronic lung condition causing breathing difficulties. Managed with inhalers and avoiding triggers."
        
        elif 'headache' in query_lower:
            return "Headaches can be tension, migraine, or cluster types. Often treated with rest, hydration, and pain relievers."
        
        elif 'fever' in query_lower:
            return "Fever is elevated body temperature, usually from infection. Rest and fluids help. See doctor if high or prolonged."
        
        elif any(word in query_lower for word in ['cold', 'flu']):
            return "Colds and flu are respiratory infections. Symptoms include cough, fever, and fatigue. Rest and fluids are important."
        
        elif 'blood pressure' in query_lower:
            return "High blood pressure often has no symptoms. Managed with diet, exercise, and medication. Regular monitoring is important."
        
        elif any(word in query_lower for word in ['allerg', 'sneez']):
            return "Allergies are immune responses to substances. Symptoms include sneezing, itching, and rashes. Antihistamines can help."
        
        else:
            return "I can provide concise medical information. Please ask about specific conditions or symptoms."
    
    def _check_medical_emergency(self, query):
        """Check for emergency medical situations"""
        query_lower = query.lower()
        
        emergency_conditions = {
            'chest pain': "🚨 CHEST PAIN could indicate a heart attack. Call emergency services immediately!",
            'heart attack': "🚨 HEART ATTACK: Call emergency services now! Symptoms: chest pain, shortness of breath.",
            'stroke': "🚨 STROKE: Remember FAST - Face drooping, Arm weakness, Speech difficulty. Call emergency services!",
            'difficulty breathing': "🚨 BREATHING DIFFICULTY: This is a medical emergency! Call for help immediately!",
            'severe bleeding': "🚨 SEVERE BLEEDING: Apply pressure and call emergency services!",
            'unconscious': "🚨 UNCONSCIOUS: Check breathing, call emergency services!",
            'suicide': "🚨 Call emergency services or a crisis helpline immediately! Your life matters!",
            'kill myself': "🚨 Call for help now! Emergency services are available 24/7!",
            'choking': "🚨 CHOKING: If can't breathe, call emergency services!",
            'severe allergic reaction': "🚨 SEVERE ALLERGIC REACTION: This can be life-threatening! Call emergency services!"
        }
        
        for condition, response in emergency_conditions.items():
            if condition in query_lower:
                return response + "\n\n📞 Call emergency services!"
        
        return None

print("✅ MedicalChatbot class defined successfully!")