from groq import Groq
import os
import time
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def query_groq(context, question=""):
    prompt = f"""
You are a financial assistant. ONLY use the below context to answer the question.
If not found, say "I don’t know based on the provided data."

Context:
{context}

Q: {question}
A:"""

    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You're an accurate, honest financial analyst. Respond clearly."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"⚠️ GROQ API error (attempt {attempt+1}/3): {e}")
            time.sleep(3)

    # Final fallback
    return "I don’t know based on the provided data."
