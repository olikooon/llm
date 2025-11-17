import os
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from pypdf import PdfReader

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

PERSIST_DIR = "./chroma_store"

vectorstore = Chroma(
    collection_name="course_slides",
    persist_directory=PERSIST_DIR,
    embedding_function=EMBEDDING_MODEL,
)

cache = {}


def extract_slides_from_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    slides = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and len(text.strip()) > 20:
            slides.append({"page": i + 1, "text": text.strip()})
    return slides


def load_lectures_to_chroma():
    lectures_dir = "./lectures"
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_docs = []
    for filename in sorted(os.listdir(lectures_dir)):
        if not filename.endswith(".pdf"):
            continue
        path = os.path.join(lectures_dir, filename)
        print(f"Reading: {filename}")
        slides = extract_slides_from_pdf(path)
        for slide in slides:
            chunks = splitter.split_text(slide["text"])
            for chunk in chunks:
                all_docs.append(chunk)

    if all_docs:
        print(f"Added {len(all_docs)} fragments to ChromaDB")
        vectorstore.add_texts(all_docs)
    else:
        print("No documents found.")


# TODO: load only 1 time at start
# load_lectures_to_chroma()


async def get_answer_from_rag(question: str) -> str:
    if question in cache:
        return cache[question]

    results = vectorstore.similarity_search(question, k=3)
    context_text = "\n\n".join([r.page_content for r in results])

    prompt = f"""
You are a teaching assistant. Answer only based on the lecture notes.
If the answer isn't in the lecture notes, say, "This information isn't in the lecture notes."

Context:
{context_text}

Question: {question}
Answer:
    """

    try:
        response = llm.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error with LLM: {e}"

    cache[question] = answer
    return answer


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Ask questions ")


@dp.message(Command("clearcache"))
async def cmd_clear_cache(message: types.Message):
    cache.clear()
    await message.answer("Cache cleared")


@dp.message(F.text)
async def handle_question(message: types.Message):
    question = message.text.strip()
    await message.answer("wait...wait..")

    answer = await get_answer_from_rag(question)
    await message.answer(answer)


async def main():
    print("Bot running")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
