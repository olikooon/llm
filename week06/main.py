import os
import sqlite3

import faiss
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def create_mock_db():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            department TEXT,
            salary REAL
        )
    """)
    data = [
        ("Alice", "marketing", 100),
        ("Bob", "engineering", 120),
        ("Carol", "marketing", 80),
        ("Dan", "sales", 70),
        ("Eve", "marketing", 90)
    ]
    cur.executemany("INSERT INTO employees (name, department, salary) VALUES (?, ?, ?)", data)
    conn.commit()
    return conn


def gemini_text_to_sql(query: str, schema: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
You are an AI that converts natural language questions into SQL queries.
All departments start with a lowercase
Use this table schema:
{schema}

Question:
{query}

Return ONLY the SQL query (no explanation, no markdown).
"""
    response = model.generate_content(prompt)
    sql_query = response.text.strip().replace("```sql", "").replace("```", "").replace(";", " COLLATE NOCASE;")

    return sql_query


def gemini_generate_context(query: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
You are a helpful assistant. The user asked:
"{query}"
Generate a concise, factual answer even if no database is available.
"""
    response = model.generate_content(prompt)
    return response.text.strip()


class Memory:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.dim = 384
        self.index = faiss.IndexFlatL2(self.dim)
        self.memory_texts = []

    def add(self, text: str):
        emb = self.embedder.encode([text])
        self.index.add(np.array(emb, dtype=np.float32))
        self.memory_texts.append(text)

    def search(self, query: str, top_k=1):
        if len(self.memory_texts) == 0:
            return None
        q_emb = self.embedder.encode([query])
        D, I = self.index.search(np.array(q_emb, dtype=np.float32), top_k)
        return self.memory_texts[I[0][0]], D[0][0]


def route_query(query: str):
    if any(word in query.lower() for word in ["salary", "department", "average", "count", "sum"]):
        return "database"
    else:
        return "predict"


def modular_rag_pipeline(user_query: str, conn, memory: Memory):
    schema = "employees(id, name, department, salary)"

    route = route_query(user_query)
    print(f"Routing decision: {route.upper()}")

    memory_result = memory.search(user_query)
    if memory_result and memory_result[1] < 0.5:
        print("Found similar query in memory!")
        return memory_result[0]

    if route == "database":
        sql_query = gemini_text_to_sql(user_query, schema)
        print(f"\nGenerated SQL:\n{sql_query}")

        cur = conn.cursor()
        cur.execute(sql_query)
        result = cur.fetchone()
        answer = f"The result is {result[0]}" if result else "No data found"
    else:
        answer = gemini_generate_context(user_query)

    memory.add(f"{user_query} -> {answer}")

    return answer


if __name__ == "__main__":
    conn = create_mock_db()
    memory = Memory()

    while True:
        query = input("\nEnter your query (or 'exit'): ")
        if query.lower() == "exit":
            break

        response = modular_rag_pipeline(query, conn, memory)
        print(f"\nResponse: {response}")
