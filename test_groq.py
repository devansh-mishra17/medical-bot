import os
from dotenv import load_dotenv

load_dotenv()

def test_groq():
    print("🧪 Testing Groq API with current models...")
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ No GROQ_API_KEY found in .env file")
        return False
    
    if not groq_key.startswith('gsk_'):
        print("❌ Invalid Groq API key format")
        return False
    
    print("✅ Valid Groq API key found")
    
    # Current available models (as of Nov 2024)
    model_names = [
        "llama-3.1-8b-instant",      # Newest fast model
        "llama-3.1-70b-versatile",   # Newest powerful model
        "llama-3.2-1b-preview",      # Lightweight option
        "llama-3.2-3b-preview",      # Balanced option
        "llama-3.2-90b-preview",     # Most powerful
        "llama-3.3-70b-specdec",     # Specialized model
        "llama-guard-3-8b",          # Safety model
        "mixtral-8x7b-32768",        # Alternative (might still work for some)
    ]
    
    for model_name in model_names:
        try:
            from langchain_groq import ChatGroq
            
            print(f"🔄 Trying model: {model_name}")
            
            llm = ChatGroq(
                groq_api_key=groq_key,
                model_name=model_name,
                temperature=0.1
            )
            
            # Test query
            response = llm.invoke("Hello! Are you working? Reply with 'YES' if successful.")
            print(f"✅ {model_name} - SUCCESS: {response.content[:50]}...")
            return model_name
            
        except Exception as e:
            error_msg = str(e)
            if 'decommissioned' in error_msg:
                print(f"❌ {model_name} - DEPRECATED")
            elif 'not exist' in error_msg:
                print(f"❌ {model_name} - NOT FOUND")
            else:
                print(f"❌ {model_name} - FAILED: {error_msg[:100]}...")
            continue
    
    print("❌ All models failed")
    print("💡 Check available models at: https://console.groq.com/playground")
    return False

if __name__ == "__main__":
    working_model = test_groq()
    if working_model:
        print(f"🎉 Use this model: {working_model}")
    else:
        print("💡 Manually check available models at: https://console.groq.com/playground")