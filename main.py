import asyncio
import aiohttp
from ollamafreeapi import OllamaFreeAPI
import fitz
import os
import re
import hashlib

client = OllamaFreeAPI()

from deep_translator import GoogleTranslator

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

from sentence_transformers import SentenceTransformer, util


BOT_TOKEN = "8637689023:AAFEtycii74oseixEM8Co0ci2TbNKcHNdfc"
S2_API_KEY = "s2k-2X83R5watwfwNkYUPa8anA0102Q8HpWOt0Epx0VR"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

model = SentenceTransformer("all-MiniLM-L6-v2")

last_results = []
pending_queries = {}

SEMANTIC_API = "https://api.semanticscholar.org/graph/v1/paper/search"


def translate_to_en(text: str) -> str:
    return GoogleTranslator(source="ru", target="en").translate(text)


def translate_to_ru(text: str) -> str:
    return GoogleTranslator(source="en", target="ru").translate(text)


def refine_query(query):
    prompt = f"""
You are a scientific information extraction system.
Extract key search terms from the query.
Replace the terms with scientific ones that are used in scientific articles, the goal is to form the most correct query in English.
Send only the final  query as a response.
RULES:
- Keep all materials, numbers, units, and properties
- DO NOT remove numeric values 
- Output ONLY a space-separated search query

User query:
{query}

"""

    response = client.chat(
        model="llama3.2:3b",
        prompt=prompt,
        temperature=0.2
    )

    return response.strip()

PDF_CACHE = "pdf_cache"
os.makedirs(PDF_CACHE, exist_ok=True)


def pdf_path_from_doi(doi):
    h = hashlib.md5(doi.encode()).hexdigest()
    return f"{PDF_CACHE}/{h}.pdf"


def extract_text(pdf_path):

    try:
        doc = fitz.open(pdf_path)

        text = ""

        for page in doc:
            text += page.get_text()

        return text

    except:
        return ""


def extract_abstract(text):

    patterns = [
        r"abstract\s*(.*?)\n\s*(introduction|keywords)",
    ]

    for p in patterns:

        m = re.search(p, text, re.IGNORECASE | re.DOTALL)

        if m:
            return m.group(1)[:1500]

    return ""


def extract_conclusion(text):

    patterns = [
        r"conclusion[s]?\s*(.*?)\n\s*(references|acknowledgment|appendix)",
        r"discussion and conclusion[s]?\s*(.*?)\n\s*(references|acknowledgment)"
    ]

    for p in patterns:

        m = re.search(p, text, re.IGNORECASE | re.DOTALL)

        if m:
            return m.group(1)[:2000]

    return ""


def extract_sections_from_pdf(pdf_path):

    try:
        doc = fitz.open(pdf_path)

        text = ""

        # первые 3 страницы
        first_pages = min(3, len(doc))
        for i in range(first_pages):
            text += doc[i].get_text()

        # последние 3 страницы
        last_pages = max(len(doc) - 3, 0)
        for i in range(last_pages, len(doc)):
            text += doc[i].get_text()

        abstract = extract_abstract(text)
        conclusion = extract_conclusion(text)

        return abstract, conclusion

    except:
        return "", ""

async def download_pdf(session, doi):

    if not doi:
        return None

    path = pdf_path_from_doi(doi)

    if os.path.exists(path):
        return path

    url = f"https://sci-hub.box/{doi}"

    try:

        async with session.get(url, timeout=20) as r:

            if r.status != 200:
                return None

            data = await r.read()

            with open(path, "wb") as f:
                f.write(data)

            return path

    except:
        return None

async def enrich_with_pdf(papers):

    async with aiohttp.ClientSession() as session:

        tasks = []

        for p in papers:
            tasks.append(download_pdf(session, p["doi"]))

        pdf_paths = await asyncio.gather(*tasks)

    for paper, path in zip(papers, pdf_paths):

        if not path:
            continue

        abstract, conclusion = extract_sections_from_pdf(path)

        if abstract:
            paper["summary"] = abstract

        paper["conclusion"] = conclusion



async def search_papers(query: str):

    headers = {
        "x-api-key": S2_API_KEY
    }

    params = {
        "query": query,
        "limit": 50,
        "fields": "title,abstract,year,authors,url,externalIds"
    }

    async with aiohttp.ClientSession() as session:

        async with session.get(
            SEMANTIC_API,
            params=params,
            headers=headers
        ) as response:

            if response.status != 200:
                print(await response.text())
                return []

            data = await response.json()

    papers = []

    for p in data.get("data", []):

        title = p.get("title") or ""
        summary = p.get("abstract") or ""
        year = str(p.get("year") or "N/A")
        url = p.get("url") or ""

        doi = None
        if p.get("externalIds"):
            doi = p["externalIds"].get("DOI")

        authors = []
        if p.get("authors"):
            authors = [a.get("name") for a in p["authors"] if a.get("name")]

        papers.append({
            "title": title,
            "summary": summary,
            "year": year,
            "url": url,
            "authors": authors,
            "doi": doi,
            "conclusion": "",
            "score": 0
        })

    return papers

def rank_papers(query: str, papers):

    texts = []

    for p in papers:

        title = p["title"] or ""
        abstract = p["summary"] or ""
        conclusion = p.get("conclusion") or ""

        text = f"{title} {title} {abstract} {conclusion}"

        texts.append(text)

    if not texts:
        return []

    query_embedding = model.encode(query, convert_to_tensor=True)

    paper_embeddings = model.encode(texts, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, paper_embeddings)[0]

    for i, paper in enumerate(papers):
        paper["score"] = float(scores[i])

    papers.sort(key=lambda x: x["score"], reverse=True)

    return papers[:5]


def apply_filters(papers, filter_text: str):

    filters = filter_text.split()
    filtered = papers

    for f in filters:

        if f.startswith("Дата>"):
            year = int(f.split(">")[1])
            filtered = [p for p in filtered if p["year"].isdigit() and int(p["year"]) > year]

        elif f.startswith("Дата<"):
            year = int(f.split("<")[1])
            filtered = [p for p in filtered if p["year"].isdigit() and int(p["year"]) < year]

        elif f.startswith("Релевантность>"):
            score = float(f.split(">")[1])
            filtered = [p for p in filtered if p["score"] > score]

        elif f.startswith("Автор:"):
            name = f.split(":")[1].lower()
            filtered = [
                p for p in filtered
                if any(name in a.lower() for a in p["authors"])
            ]

    return filtered


@dp.message(Command("start"))
async def start_command(message: types.Message):

    await message.answer(
        "Привет 👋\n\n"
        "Я бот для поиска научных статей.\n\n"
        "Команды:\n"
        "/search <запрос>\n"
        "/confirm\n"
        "/filter\n\n"
        "Пример:\n"
        "/search машинное обучение"
    )


@dp.message(Command("search"))
async def search_command(message: types.Message):

    parts = message.text.split(maxsplit=1)

    if len(parts) < 2:
        await message.answer("Напиши запрос после /search")
        return

    query_ru = parts[1]

    query_en = translate_to_en(query_ru)

    refined_query = refine_query(query_ru)

    pending_queries[message.from_user.id] = refined_query

    await message.answer(
        "🧠 Я проанализировал ваш запрос.\n\n"
        f"Исходный запрос:\n{query_ru}\n\n"
        f"Уточнённый запрос:\n{refined_query}\n\n"
        "Если всё верно — напишите /confirm"
    )


@dp.message(Command("confirm"))
async def confirm_query(message: types.Message):

    user_id = message.from_user.id

    if user_id not in pending_queries:
        await message.answer("Нет запроса для подтверждения")
        return

    query = pending_queries[user_id]

    await message.answer("🔎 Ищу статьи...")

    papers = await search_papers(query)

    await message.answer("📄 Загружаю PDF и анализирую статьи...")

    await enrich_with_pdf(papers)


    if not papers:
        await message.answer("Ничего не найдено")
        return

    ranked = rank_papers(query, papers)

    global last_results
    last_results = ranked

    result = "📚 Лучшие статьи:\n\n"

    for paper in ranked:

        title_ru = translate_to_ru(paper["title"])
        summary = paper["summary"] or ""
        summary_ru = ""
        if summary:
            summary_ru = translate_to_ru(summary)
            summary_ru = summary_ru[:500]
        authors = ", ".join(paper["authors"][:3])
        score = round(paper["score"], 3)

        result += (
            f"📄 {title_ru}\n"
            f"👨‍🔬 {authors}\n"
            f"📅 {paper['year']}\n"
            f"🎯 Релевантность: {score}\n"
            f"🧠 {summary_ru}\n"
            f"🔗 {paper['url']}\n\n"
        )

    await message.answer(result)

    del pending_queries[user_id]


@dp.message(Command("filter"))
async def filter_command(message: types.Message):

    global last_results

    if not last_results:
        await message.answer("Сначала выполните поиск")
        return

    parts = message.text.split(maxsplit=1)

    if len(parts) < 2:
        await message.answer(
            "Пример:\n"
            "/filter Дата>2018 Релевантность>0.4 Автор:Smith"
        )
        return

    filter_text = parts[1]

    filtered = apply_filters(last_results, filter_text)

    if not filtered:
        await message.answer("Ничего не найдено")
        return

    result = "📚 Отфильтрованные статьи:\n\n"

    for paper in filtered[:5]:

        title_ru = translate_to_ru(paper["title"])
        summary = paper["summary"] or ""
        summary_ru = translate_to_ru(summary[:300]) if summary else ""
        authors = ", ".join(paper["authors"][:3])
        score = round(paper["score"], 3)

        result += (
            f"📄 {title_ru}\n"
            f"👨‍🔬 {authors}\n"
            f"📅 {paper['year']}\n"
            f"🎯 Релевантность: {score}\n"
            f"🧠 {summary_ru}...\n"
            f"🔗 {paper['url']}\n\n"
        )

    await message.answer(result)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())