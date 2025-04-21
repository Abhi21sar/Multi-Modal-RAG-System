# app/generator.py

from openai import OpenAI
from app.embedder import get_text_embedding

# Set your OpenAI API key

client = OpenAI(api_key="Your_OPENAI_API_KEY") 

def build_prompt(context_docs, user_query):
    context = "\n\n".join([doc["content"][:1000] for doc in context_docs])
    prompt = f"""
You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context}

Question: {user_query}
Answer:"""
    return prompt

def get_answer_from_gpt(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

def generate_answer(user_query, vector_store, top_k=3):
    query_vector = get_text_embedding(user_query)
    retrieved_docs = vector_store.search(query_vector, top_k=top_k)
    prompt = build_prompt(retrieved_docs, user_query)
    answer = get_answer_from_gpt(prompt)
    return answer, retrieved_docs