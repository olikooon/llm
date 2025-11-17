import csv
import json
import os
import re
import time

import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

LECTURES_DIR = "./lectures"
OUTPUT_FILE = "lecture_questions.csv"


def extract_slides_from_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    slides = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and len(text.strip()) > 20:
            slides.append({"page": i + 1, "text": text.strip()})
    return slides


def generate_question_from_slide(slide_text: str, lecture_name: str, slide_num: int):
    prompt = f"""
Create a multiple-choice question from this lecture slide content.

Output must be in this JSON format:
{{
  "id": "{lecture_name}_{slide_num}",
  "question": "...",
  "choices": {{
    "text": ["...", "...", "...", "..."],
    "label": ["A", "B", "C", "D"]
  }},
  "answerKey": "A"
}}

Slide:
\"\"\"{slide_text}\"\"\"
"""

    retries = 3
    for attempt in range(retries):
        try:
            response = llm.generate_content(prompt)
            text = response.text.strip()

            match = re.search(r'\{[\s\S]*\}', text)

            qa = json.loads(match.group(0))
            return qa

        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "exceeded" in err.lower():
                wait = 60
                time.sleep(wait)
                continue

            return None
    return None


def build_dataset():
    dataset = []
    for filename in sorted(os.listdir(LECTURES_DIR)):
        if not filename.lower().endswith(".pdf"):
            continue

        print(f"Reading {filename}")
        lecture_name = filename.replace(".pdf", "")
        slides = extract_slides_from_pdf(os.path.join(LECTURES_DIR, filename))

        for slide in slides:
            qa = generate_question_from_slide(slide["text"], lecture_name, slide["page"])
            if qa:
                dataset.append(qa)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "choices", "answerKey"])
        writer.writeheader()
        for qa in dataset:
            writer.writerow({
                "id": qa.get("id", ""),
                "question": qa.get("question", ""),
                "choices": json.dumps(qa.get("choices", {}), ensure_ascii=False),
                "answerKey": qa.get("answerKey", "")
            })


if __name__ == "__main__":
    build_dataset()
