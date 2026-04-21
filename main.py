import asyncio
import aiohttp
import xml.etree.ElementTree as ET

from deep_translator import GoogleTranslator

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

from sentence_transformers import SentenceTransformer, util


last_results = []
model = SentenceTransformer("all-MiniLM-L6-v2")

def translate_to_en(text: str) -> str:
    return GoogleTranslator(source='ru', target='en').translate(text)

def translate_to_ru(text: str) -> str:
    return GoogleTranslator(source='en', target='ru').translate(text)

bot = Bot(token="8637689023:AAFEtycii74oseixEM8Co0ci2TbNKcHNdfc")
dp = Dispatcher()

ARXIV_API = "http://export.arxiv.org/api/query"

@dp.message(Command("start"))
async def start_command(message: types.Message):

    await message.answer(
        "Привет 👋\n\n"
        "Я бот для поиска научных статей.\n\n"
        "Используй:\n"
        "/search <запрос>\n\n"
        "Например:\n"
        "/search Машинное обучение"
    )


async def search_papers(query: str):

    headers = {
        "User-Agent": "research-bot"
    }

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": 50
    }

    async with aiohttp.ClientSession() as session:

        async with session.get(
            ARXIV_API,
            params=params,
            headers=headers,
            ssl=False
        ) as response:

            print("STATUS:", response.status)

            if response.status != 200:
                print(await response.text())
                return []

            text = await response.text()

    root = ET.fromstring(text)

    ns = {
        "atom": "http://www.w3.org/2005/Atom"
    }

    papers = []

    for entry in root.findall("atom:entry", ns):

        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        published = entry.find("atom:published", ns)
        link = entry.find("atom:id", ns)

        authors = entry.findall("atom:author/atom:name", ns)

        papers.append({
            "title": title.text if title is not None else "",
            "summary": summary.text if summary is not None else "",
            "year": published.text[:4] if published is not None else "N/A",
            "url": link.text if link is not None else "",
            "authors": [a.text for a in authors],
            "score": 0
        })

    return papers


def rank_papers(query: str, papers):

    texts = []

    for p in papers:
        texts.append(p["title"] + " " + p["title"] + " " + p["summary"])

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


        if f.startswith("Дата публикации>"):
            year = int(f.split(">")[1])
            filtered = [
                p for p in filtered
                if p["year"].isdigit() and int(p["year"]) > year
            ]


        elif f.startswith("Дата публикации<"):
            year = int(f.split("<")[1])
            filtered = [
                p for p in filtered
                if p["year"].isdigit() and int(p["year"]) < year
            ]


        elif f.startswith("Релевантность>"):
            score = float(f.split(">")[1])
            filtered = [
                p for p in filtered
                if p["score"] > score
            ]

        elif f.startswith("Автор:"):
            name = f.split(":")[1].lower()

            filtered = [
                p for p in filtered
                if any(name in a.lower() for a in p["authors"])
            ]

    return filtered

@dp.message(Command("search"))
async def search_command(message: types.Message):

    parts = message.text.split(maxsplit=1)

    if len(parts) < 2:

        await message.answer("Напиши запрос после /search")
        return

    query_ru = parts[1]
    query_en = translate_to_en(query_ru)
    await message.answer("🔎 Ищу статьи...")
    papers = await search_papers(query_en)

    if not papers:
        await message.answer("Ничего не найдено")
        return

    ranked = rank_papers(query_en, papers)
    global last_results
    last_results = ranked
    result = "📚 Лучшие статьи:\n\n"

    for paper in ranked:

        title_ru = translate_to_ru(paper["title"])
        summary_ru = translate_to_ru(paper["summary"][:300])
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

@dp.message(Command("filter"))
async def filter_command(message: types.Message):

    global last_results

    if not last_results:
        await message.answer("Сначала сделай поиск через /search")
        return

    parts = message.text.split(maxsplit=1)

    if len(parts) < 2:
        await message.answer(
            "Пример фильтрации:\n"
            "/filter Дата публикации>2000, Дата публикации<2000, Релевантность>0.4 Автор:Conor Mc Keever"
        )
        return

    filter_text = parts[1]
    filtered = apply_filters(last_results, filter_text)

    if not filtered:
        await message.answer("Ничего не найдено после фильтрации")
        return

    result = "📚 Отфильтрованные статьи:\n\n"

    for paper in filtered[:5]:

        title_ru = translate_to_ru(paper["title"])
        summary_ru = translate_to_ru(paper["summary"][:300])
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