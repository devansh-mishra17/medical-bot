from langchain_huggingface import HuggingFaceEmbedding
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os

class MedicalChatbot:
    def __init__(self):
        self.initialize_components()
    
    def initialize_components(self):
        print("🔄 Initializing Medical Chatbot with LangChain 1.0.7...")
        
        # Create comprehensive medical knowledge
        medical_knowledge = [
            "Headaches can be caused by stress, dehydration, or eye strain. Rest and hydration often help. Over-the-counter pain relievers like ibuprofen can provide relief.",
            "Fever is a common symptom of infection. Normal body temperature is 98.6°F. Rest, fluids, and fever reducers like acetaminophen can help manage fever.",
            "Common cold symptoms include runny nose, sore throat, and coughing. Treatment focuses on rest, hydration, and over-the-counter cold medications.",
            "Back pain can be alleviated with proper posture, stretching exercises, and over-the-counter pain relievers. Physical therapy may help chronic cases.",
            "Allergy symptoms include sneezing, itchy eyes, and runny nose. Antihistamines can provide relief. Avoid known allergens when possible.",
            "Healthy diet includes fruits, vegetables, whole grains, and lean proteins. Limit processed foods and sugar for better health.",
            "Regular exercise for 30 minutes daily improves cardiovascular health and reduces stress. Include both cardio and strength training.",
            "For minor cuts, clean with soap and water, apply antibiotic ointment, and cover with a bandage. Watch for signs of infection.",
            "High blood pressure management includes reducing salt intake, regular exercise, and taking prescribed medications as directed.",
            "Diabetes care involves monitoring blood sugar, eating balanced meals, regular exercise, and taking medications as prescribed.",
            "Stress management techniques include deep breathing, meditation, regular exercise, and maintaining work-life balance.",
            "Good sleep hygiene includes consistent sleep schedule, dark quiet environment, and avoiding screens before bedtime.",
            "Indigestion and heartburn can be relieved by eating smaller meals and avoiding spicy foods before lying down.",
            "Muscle pain often responds to rest, ice packs, and anti-inflammatory medications. Gentle stretching prevents stiffness.",
            "Seasonal allergies can be managed with allergy medications and keeping windows closed during high pollen counts."
        ]
        
        # Create documents
        documents = [Document(page_content=text) for text in medical_knowledge]
        
        # Initialize embeddings - this should work with your setup
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        print(f"✅ Medical knowledge base created with {len(medical_knowledge)} facts!")
    
    def get_response(self, user_message):
        # Emergency check first
        emergency_response = self._check_medical_emergency(user_message)
        if emergency_response:
            return emergency_response
        
        try:
            # Use similarity search - this is the core functionality
            docs = self.vectorstore.similarity_search(user_message, k=2)
            
            if docs:
                # Format a nice response
                response = self._format_medical_response(docs, user_message)
            else:
                response = self._get_fallback_response(user_message)
            
            # Always add medical disclaimer
            return response + "\n\n---\n⚠️ **Medical Disclaimer**: I am an AI assistant. Always consult healthcare professionals for medical advice."
            
        except Exception as e:
            print(f"Error in medical response: {e}")
            return "I can provide general health information. For medical advice, please consult a healthcare professional."
    
    def _format_medical_response(self, docs, user_question):
        """Create a well-formatted response from documents"""
        main_info = docs[0].page_content
        
        # Add additional relevant info if available
        if len(docs) > 1:
            additional_info = f"\n\nAdditional information: {docs[1].page_content}"
            return main_info + additional_info
        else:
            return main_info
    
    def _get_fallback_response(self, query):
        """Provide helpful responses for common medical questions"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['headache', 'head pain']):
            return "Headaches can have various causes. Rest in a quiet room, stay hydrated, and consider over-the-counter pain relief. See a doctor for severe or frequent headaches."
        
        elif any(word in query_lower for word in ['fever', 'temperature']):
            return "Fever helps your body fight infection. Rest and drink plenty of fluids. Contact a doctor if fever is over 103°F or lasts more than 3 days."
        
        elif any(word in query_lower for word in ['cold', 'flu', 'cough']):
            return "For cold and flu: get plenty of rest, drink fluids, and use over-the-counter medications for symptom relief. See a doctor if symptoms are severe."
        
        elif any(word in query_lower for word in ['pain', 'hurt']):
            return "Pain management depends on the cause. Rest the affected area and use over-the-counter pain relievers as directed. Seek medical help for severe pain."
        
        else:
            return "I can provide general health information. For specific medical concerns, please consult with a healthcare professional."
    
    def _check_medical_emergency(self, query):
        """Check for emergency situations that need immediate help"""
        query_lower = query.lower()
        
        emergencies = {
            'chest pain': "🚨 CHEST PAIN could be a heart attack. Call emergency services immediately!",
            'heart attack': "🚨 HEART ATTACK: Call emergency services now! Symptoms include chest pain and shortness of breath.",
            'stroke': "🚨 STROKE: Call emergency services! Look for face drooping, arm weakness, speech difficulty.",
            'difficulty breathing': "🚨 BREATHING PROBLEMS: This is an emergency! Call for help immediately!",
            'severe bleeding': "🚨 SEVERE BLEEDING: Apply pressure and call emergency services!",
            'unconscious': "🚨 UNCONSCIOUS person: Check breathing and call emergency services!",
            'suicide': "🚨 Please call emergency services or a crisis helpline immediately! Your life matters!",
            'kill myself': "🚨 Call for help now! Emergency services and crisis lines are available 24/7!"
        }
        
        for emergency, response in emergencies.items():
            if emergency in query_lower:
                return response + "\n\n📞 Call your local emergency number RIGHT NOW!"
        
        return None