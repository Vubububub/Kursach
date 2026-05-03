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

model = SentenceTransformer("intfloat/e5-base-v2")

last_results = []
pending_queries = {}

SEMANTIC_API = "https://api.semanticscholar.org/graph/v1/paper/search"


def translate_to_en(text: str) -> str:
    return GoogleTranslator(source="ru", target="en").translate(text)


def translate_to_ru(text: str) -> str:
    return GoogleTranslator(source="en", target="ru").translate(text)

def llm_filter_papers(papers, filter_query):

    papers_text = ""

    for i, p in enumerate(papers):

        title = p["title"]
        year = p["year"]
        abstract = (p["summary"] or "")[:400]

        papers_text += f"[{i}] {title} | {year}\n{abstract}\n\n"

    prompt = f"""
You are a scientific paper filtering system.

User filter request:
{filter_query}

Below is a list of papers.

Select the papers that satisfy the filter request.

Return ONLY paper indexes separated by commas.

Example output:
1,4,7,10

Papers:
{papers_text}
"""

    response = client.chat(
        model="llama3.2:3b",
        prompt=prompt,
        temperature=0
    )

    response = response.strip()

    indexes = []

    for x in response.replace(" ", "").split(","):
        if x.isdigit():
            i = int(x)
            if 0 <= i < len(papers):
                indexes.append(i)

    return [papers[i] for i in indexes]



def refine_query(query):
    prompt = f"""
You are a scientific information extraction system.
Extract key search terms from the query.
Replace the terms with scientific ones that are used in scientific articles, the goal is to form the most correct query IN ENGLISH.
Send only the final  query as a response.
RULES:
- Keep all materials, numbers, units, and properties
- DO NOT remove numeric values 
- Output ONLY a space-separated search query.
- DO NOT USE "becomes:", "Query:"

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


        authors = [
            a.get("name")
            for a in p.get("authors", [])
            if a.get("name")
        ]

        # DOI
        doi = (p.get("externalIds") or {}).get("DOI")

        #
        fields = p.get("fieldsOfStudy") or []


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


            "score": 0
        })

    return papers

def rank_papers(query: str, papers):

    def extract_keywords(q):
        import re
        tokens = re.findall(r"[a-zA-Z0-9\./\-]+", q.lower())

        out = []
        for t in tokens:
            out.extend(t.split("/"))
            out.extend(t.split("-"))

        return set([x for x in out if len(x) > 1])


    keywords = extract_keywords(query)

    texts = []
    for p in papers:
        text = (
            (p.get("title") or "") + " " +
            (p.get("summary") or "")
        ).lower()
        texts.append(text)


    query_emb = model.encode("query: " + query, convert_to_tensor=True)
    paper_emb = model.encode(
        ["passage: " + t for t in texts],
        convert_to_tensor=True
    )

    sims = util.cos_sim(query_emb, paper_emb)[0]

    for i, p in enumerate(papers):

        text = texts[i]

        score = float(sims[i])


        hits = 0

        for kw in keywords:

            if kw in text:
                hits += 1


        keyword_score = min(hits / max(len(keywords), 1), 1.0)


        score = 0.85 * score + 0.15 * keyword_score

        p["score"] = score

    papers.sort(key=lambda x: x["score"], reverse=True)

    return papers[:5]



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
    global last_results
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

    last_results = papers

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
            "/filter за последние 5 лет\n"

        )
        return

    filter_text = parts[1]

    await message.answer("🤖 Анализирую статьи через LLM...")

    filtered = llm_filter_papers(last_results, filter_text)

    if not filtered:
        await message.answer("Ничего не найдено")
        return

    result = "📚 Отфильтрованные статьи:\n\n"

    for paper in filtered[:5]:

        title_ru = translate_to_ru(paper["title"])

        summary = paper["summary"] or ""
        summary_ru = translate_to_ru(summary[:300]) if summary else ""

        authors = ", ".join(paper["authors"][:3])

        result += (
            f"📄 {title_ru}\n"
            f"👨‍🔬 {authors}\n"
            f"📅 {paper['year']}\n"
            f"🧠 {summary_ru}...\n"
            f"🔗 {paper['url']}\n\n"
        )

    await message.answer(result)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())