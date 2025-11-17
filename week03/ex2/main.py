import os
import asyncio
import logging
from typing import Tuple, List, Dict
import requests
from dotenv import load_dotenv
import time

from readability import Document
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import google.generativeai as genai
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArticleProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.session = self._create_session()
        
    def _create_session(self):
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        return session
        
    def scrape_article(self, url: str) -> str:
        try:
            logger.info(f"Scraping URL: {url}")
            
            time.sleep(1)
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"Non-HTML content type: {content_type}")
                return ""
            
            doc = Document(response.text)
            
            title = doc.title()
            cleaned_content = self._clean_html(doc.summary())
            
            if not cleaned_content.strip():
                logger.warning("No content extracted after cleaning")
                return ""
                
            full_content = f"{title}\n\n{cleaned_content}"
            logger.info(f"Successfully scraped article: {title[:50]}...")
            
            return full_content
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} for {url}: {e}")
            if e.response.status_code == 403:
                return self._fallback_scraping(url)
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")
            return ""
    
    def _fallback_scraping(self, url: str) -> str:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            doc = Document(response.text)
            return self._clean_html(doc.summary())
        except Exception as e:
            logger.error(f"Fallback scraping also failed: {e}")
            return ""
    
    def _clean_html(self, html_content: str) -> str:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            cleaned_text = '\n'.join(lines)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            return html_content
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        if not text:
            return []
            
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks


class VectorDatabase:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="articles")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def store_article(self, url: str, content: str):
        processor = ArticleProcessor()
        chunks = processor.chunk_text(content)

        if not chunks:
            return

        embeddings = self.embedding_model.encode(chunks).tolist()

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"url": url, "chunk_id": i}],
                ids=[f"{url}_{i}"]
            )

    def search_similar(self, query: str, n_results: int = 3) -> List[str]:
        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        return results['documents'][0] if results['documents'] else []


class DistilBERTQA:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
        self.model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

    def answer_question(self, question: str, context: str) -> Tuple[str, float]:
        try:
            inputs = self.tokenizer(
                question,
                context,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits) + 1

            start_prob = torch.softmax(start_logits, dim=-1)[0, start_idx]
            end_prob = torch.softmax(end_logits, dim=-1)[0, end_idx - 1]
            confidence = (start_prob * end_prob).item()

            answer_tokens = inputs['input_ids'][0][start_idx:end_idx]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

            return answer, confidence

        except Exception as e:
            logger.error(f"Error in DistilBERT QA: {e}")
            return "I couldn't process that question", 0.0


class GeminiQA:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    async def answer_question(self, question: str, context: str) -> str:
        try:
            prompt = f"""
Based on the following context, please answer the question. If the context doesn't contain enough information to answer properly, please indicate that.

Context: {context}

Question: {question}

Answer:"""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )

            return response.text
        except Exception as e:
            logger.error(f"Error in Gemini QA: {e}")
            return "I couldn't process that question."


class TelegramBot:
    def __init__(self, telegram_token: str, gemini_api_key: str):
        self.bot = Bot(token=telegram_token)
        self.dp = Dispatcher()

        self.vector_db = VectorDatabase()
        self.article_processor = ArticleProcessor()
        self.distilbert_qa = DistilBERTQA()
        self.gemini_qa = GeminiQA(gemini_api_key)

        self.dp.message(Command("start"))(self.start_handler)
        self.dp.message()(self.message_handler)

    async def start_handler(self, message: Message):
        welcome_text = """
send url or ask a question
        """
        await message.answer(welcome_text)

    async def process_url(self, url: str) -> str:
        try:
            content = self.article_processor.scrape_article(url)
            if not content:
                return "I couldn't extract content from that URL."

            self.vector_db.store_article(url, content)

            return f"Article stored"

        except Exception as e:
            logger.error(f"Error processing URL: {e}")
            return "Error."

    async def process_question(self, question: str) -> str:
        try:
            similar_docs = self.vector_db.search_similar(question)
            if not similar_docs:
                return "No relevant information found."

            context = "\n".join(similar_docs[:3])

            gemini_answer = await self.gemini_qa.answer_question(question, context)
            distilbert_answer, confidence = self.distilbert_qa.answer_question(question, context)

            response = f"Question: {question}\n\n"
            response += f"Context used: {context[:200]}...\n\n"
            response += "--- Gemini Answer ---\n"
            response += f"{gemini_answer}\n\n"
            response += "--- DistilBERT Answer ---\n"
            response += f"{distilbert_answer}\n"
            response += f"(Confidence: {confidence:.2f})"

            return response

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return "Error."

    async def message_handler(self, message: Message):
        try:
            user_input = message.text.strip()

            if user_input.startswith(('http://', 'https://')):
                processing_msg = await message.answer("Processing URL...")
                response = await self.process_url(user_input)
                await processing_msg.delete()
                await message.answer(response)

            else:
                processing_msg = await message.answer("wait...")
                response = await self.process_question(user_input)
                await processing_msg.delete()

                if len(response) > 4096:
                    for i in range(0, len(response), 4096):
                        await message.answer(response[i:i + 4096])
                else:
                    await message.answer(response)

        except Exception as e:
            logger.error(f"Error in message handler: {e}")
            await message.answer("An error occurred.")

    async def run(self):
        logger.info("Starting Telegram bot...")
        await self.dp.start_polling(self.bot)


def main():
    bot = TelegramBot(BOT_TOKEN, GEMINI_API_KEY)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped")


if __name__ == "__main__":
    main()
