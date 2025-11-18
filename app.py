from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

print("🚀 Starting Medical Chatbot with Groq AI...")

# Try Groq version first for best performance
try:
    from chatbot.groq_chain import GroqMedicalChatbot as MedicalChatbot
    print("✅ Medical Chatbot with Groq AI loaded!")
except ImportError as e:
    print(f"❌ Groq version failed: {e}")
    try:
        from chatbot.working_chain import MedicalChatbot
        print("✅ Using standard Medical Chatbot")
    except ImportError:
        class BasicMedicalChatbot:
            def get_response(self, user_message):
                return "Medical chatbot available. Please consult healthcare professionals."
        MedicalChatbot = BasicMedicalChatbot
        print("✅ Using basic chatbot")

load_dotenv()

app = Flask(__name__)

# Check if Groq API key is available
groq_key = os.getenv("GROQ_API_KEY")
if groq_key and groq_key.startswith('gsk_'):
    print("🔑 Groq API key detected - Enhanced AI enabled!")
else:
    print("⚠️  No Groq API key - Using local mode")

# Initialize chatbot
print("🩺 Starting Medical Chatbot...")
chatbot = MedicalChatbot()
print("✅ Medical Chatbot Ready!")
print("🌐 Open: http://localhost:5000")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'response': 'Please enter a medical question.',
                'status': 'error'
            })
        
        response = chatbot.get_response(user_message)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'response': 'Please consult healthcare professionals for medical advice.',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)