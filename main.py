import asyncio
import aiohttp
from ollamafreeapi import OllamaFreeAPI


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




async def search_papers(query: str):

    headers = {
        "x-api-key": S2_API_KEY
    }

    params = {
        "query": query,
        "limit": 50,
        "fields": (
            "title,abstract,year,authors,url,"
            "externalIds,venue,citationCount,"
            "influentialCitationCount,fieldsOfStudy,"
            "s2FieldsOfStudy"
        )
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
        abstract = p.get("abstract") or ""

        year = p.get("year")
        year = str(year) if year else "N/A"

        url = p.get("url") or ""

        # authors
        authors = [
            a.get("name")
            for a in p.get("authors", [])
            if a.get("name")
        ]

        # DOI
        doi = (p.get("externalIds") or {}).get("DOI")

        # fieldsOfStudy
        fields = p.get("fieldsOfStudy") or []

        # s2FieldsOfStudy (очень полезно для семантики)
        s2_fields = []
        for f in p.get("s2FieldsOfStudy") or []:
            if isinstance(f, dict):
                name = f.get("category")
                if name:
                    s2_fields.append(name)

        papers.append({
            "title": title,
            "summary": abstract,
            "year": year,
            "url": url,
            "authors": authors,
            "doi": doi,

            "fields": fields,
            "s2_fields": s2_fields,

            # для ранжирования
            "score": 0
        })

    return papers

def rank_papers(query: str, papers):

    texts = []

    for p in papers:

        title = p["title"] or ""
        abstract = p["summary"] or ""

        fields = " ".join(p.get("fields", []))
        s2_fields = " ".join(p.get("s2_fields", []))

        text = (
            title + " " + title +
            abstract + " " +
            fields + " " +
            s2_fields
        )

        texts.append(text)

    query_emb = model.encode(query, convert_to_tensor=True)
    paper_emb = model.encode(texts, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, paper_emb)[0]

    for i, p in enumerate(papers):
        p["score"] = float(scores[i])

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
            f"🧠 {summary_ru}...\n"
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