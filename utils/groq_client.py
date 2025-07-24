from groq import Groq


groq_client = Groq(api_key="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")

def query_groq(context, question):
    prompt = f"""
You are a financial assistant. ONLY use the below context to answer the question.
If not found, say "I don’t know based on the provided data."

Context:
{context}

Q: {question}
A:"""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You're an accurate, honest financial analyst. Respond clearly."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
