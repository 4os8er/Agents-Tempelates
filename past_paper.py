import os
import sys
import time
import json
import re
import requests

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

SERPER_URL = "https://api.serper.live/v1/search"

MAX_SEARCH_RESULTS = 10
SEARCH_TIMEOUT = 30
MAX_SEARCH_ATTEMPTS = 2


if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY is missing from .env")
    sys.exit(1)

if not SERPER_API_KEY:
    print("❌ SERPER_API_KEY is missing from .env")
    sys.exit(1)


# ============================================================
# 2. SEARCH FUNCTION
# ============================================================

def serper_search(query: str):

    if not query or not query.strip():
        return []

    headers = {
        "Authorization": f"Bearer {SERPER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "country": "PK",
        "language": "en",
        "device": "desktop",
        "limit": MAX_SEARCH_RESULTS,
    }

    last_error = None

    for attempt in range(1, MAX_SEARCH_ATTEMPTS + 1):

        try:

            response = requests.post(
                SERPER_URL,
                headers=headers,
                json=payload,
                timeout=SEARCH_TIMEOUT,
            )

            if response.status_code == 200:

                data = response.json()

                results = data.get("organic", [])

                clean_results = []

                for item in results:

                    clean_results.append({
                        "position": item.get("position", 0),
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "domain": item.get("domain", ""),
                        "snippet": item.get("snippet", ""),
                    })

                return clean_results

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        except requests.exceptions.Timeout:

            last_error = "Search request timed out."

        except requests.exceptions.RequestException as e:

            last_error = f"Network error: {e}"

        except ValueError:

            last_error = "Invalid JSON returned by Serper."

        if attempt < MAX_SEARCH_ATTEMPTS:
            time.sleep(1)

    raise RuntimeError(
        f"Google search failed after "
        f"{MAX_SEARCH_ATTEMPTS} attempts.\n"
        f"Reason: {last_error}"
    )


# ============================================================
# 3. TEXT UTILITIES
# ============================================================

def normalize(text: str) -> str:

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s\-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def contains(text: str, value: str) -> bool:

    if not value:
        return False

    return normalize(value) in normalize(text)


# ============================================================
# 4. RESULT SCORING
# ============================================================

def score_result(result, requirements):

    combined_text = " ".join([
        result.get("title", ""),
        result.get("snippet", ""),
        result.get("domain", ""),
        result.get("url", ""),
    ])

    score = 0
    reasons = []

    exam = requirements.get("exam")
    category = requirements.get("category")
    university = requirements.get("university")
    board = requirements.get("board")
    program = requirements.get("program")
    year = requirements.get("year")
    shift = requirements.get("shift")

    if contains(combined_text, exam):
        score += 30
        reasons.append("exam matched")

    if contains(combined_text, category):
        score += 25
        reasons.append("category matched")

    if contains(combined_text, university):
        score += 20
        reasons.append("university matched")

    if contains(combined_text, board):
        score += 20
        reasons.append("board matched")

    if contains(combined_text, program):
        score += 10
        reasons.append("program matched")

    if contains(combined_text, year):
        score += 20
        reasons.append("year matched")

    if contains(combined_text, shift):
        score += 10
        reasons.append("shift matched")

    url = normalize(result.get("url", ""))

    if ".pdf" in url:
        score += 10
        reasons.append("PDF")

    return score, reasons


# ============================================================
# 5. RESULT RANKING
# ============================================================

def rank_results(results, requirements):

    ranked = []

    for result in results:

        score, reasons = score_result(
            result,
            requirements
        )

        ranked.append({
            "score": score,
            "reasons": reasons,
            "result": result,
        })

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked


# ============================================================
# 6. SEARCH TOOL FOR LANGCHAIN
# ============================================================

@tool
def live_google_search(query: str) -> str:
    """
    Search Google live using Serper for real past papers,
    answer keys, university entry tests, NTS/NAT papers,
    board papers and related educational documents.

    Use this tool whenever live web information is needed.
    Never invent URLs or papers.
    """

    try:

        results = serper_search(query)

        if not results:

            return json.dumps({
                "status": "no_results",
                "query": query,
                "results": []
            }, ensure_ascii=False)

        return json.dumps({
            "status": "success",
            "query": query,
            "results": results
        }, ensure_ascii=False)

    except Exception as e:

        return json.dumps({
            "status": "error",
            "query": query,
            "error": str(e)
        }, ensure_ascii=False)


# ============================================================
# 7. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ExamVault AI.

You are an AI agent that helps users find REAL past papers
from the live internet.

Your main goal is:

USER REQUEST
    ↓
UNDERSTAND EXACT REQUIREMENT
    ↓
ASK ONLY NECESSARY QUESTIONS
    ↓
SEARCH LIVE GOOGLE
    ↓
CHECK SEARCH RESULTS
    ↓
RANK RESULTS
    ↓
RETURN THE BEST VERIFIED LINKS

============================================================
CORE RULES
============================================================

1. NEVER invent a past paper.

2. NEVER invent a URL.

3. NEVER create a fake PDF link.

4. NEVER claim a paper is exact unless the available
   search evidence supports that conclusion.

5. If the user has not provided enough information,
   ask for the missing information.

6. NEVER ask again for information the user already gave.

7. Ask questions naturally, not like a boring form.

8. Keep questions short and easy to understand.

9. Prefer official university, testing organization,
   board or government sources.

10. If an official source is unavailable, use reliable
    educational sources.

11. Clearly distinguish:
       EXACT MATCH
       LIKELY MATCH
       RELATED RESULT

12. If the exact requested paper cannot be verified,
    honestly tell the user.

13. You can provide alternative relevant papers.

14. Never enter an infinite search loop.

15. Do not perform unnecessary searches.

16. Do not ask for shift if that exam normally
    does not have a meaningful shift distinction.

============================================================
INFORMATION TO IDENTIFY
============================================================

Depending on the request, identify:

Exam type:
- Entry Test
- NTS
- NAT
- GAT
- MDCAT
- ECAT
- NUMS
- University Entry Test
- Board Exam
- Matric
- Intermediate
- Other

Exam name:
Examples:
- NAT
- NTS
- FAST Entry Test
- COMSATS Entry Test
- ECAT
- MDCAT

Category:
Examples:
- NAT-ICS
- NAT-IE
- NAT-IM
- NAT-IA
- NAT-IGS
- etc.

University:
Examples:
- COMSATS
- FAST
- NUST
- Punjab University
- UET

Campus:
Examples:
- Lahore
- Islamabad
- Karachi
- Peshawar

Program:
Examples:
- BSCS
- BS AI
- BS Cyber Security
- Engineering

Board:
Examples:
- BISE Lahore
- BISE Rawalpindi
- FBISE

Year:
Examples:
- 2026
- 2025
- 2024
- 2020-2025
- all available

Shift:
Examples:
- Morning
- Evening
- First shift
- Second shift
- Any shift
- All shifts

Paper type:
- Original paper
- Past paper
- Solved paper
- Practice paper
- Answer key
- Paper + answer key

============================================================
SMART QUESTIONING
============================================================

Example:

User:
"I need NAT past paper."

Ask:

"Bilkul. NAT ka kaunsa category paper chahiye?
NAT-ICS, NAT-IE, NAT-IM, ya koi aur?"

Then if category is given:

"Kaunsa year chahiye?"

Then:

"Morning, evening ya koi bhi shift?"

Only ask what is actually necessary.

============================================================
MULTIPLE YEARS
============================================================

If user says:

"Give me NAT-ICS papers from 2020 to 2025."

DO NOT ask:

"2020?"
"2021?"
"2022?"

Instead understand this as a multi-year search.

Search efficiently and organize results by year.

============================================================
SEARCH STRATEGY
============================================================

When searching, make targeted queries.

Example:

"NAT ICS 2025 past paper PDF"

Then if necessary:

"NAT-ICS 2025 solved past paper answer key"

Then if necessary:

"NTS NAT ICS 2025 paper"

Do not repeat the exact same search unnecessarily.

============================================================
RESULT EVALUATION
============================================================

Search results may include:

- Actual PDF
- Educational article
- Answer key
- Forum
- Blog
- YouTube
- University page
- Fake/irrelevant page

Do NOT assume every result is a paper.

Use:
title
URL
domain
snippet

to judge relevance.

============================================================
FINAL RESPONSE
============================================================

When useful results are found, organize the response like:

Exam:
NAT-ICS

Year:
2025

Shift:
Morning

Result:

1. Exact / Best Match
Title
URL
Why it matches

2. Answer Key
Title
URL

3. Alternative
Title
URL

Verification:
Explain briefly whether the exact year/category/shift
was actually supported by the search evidence.

If exact paper was NOT found:

"Exact paper verify nahi ho saka."

Then give the closest useful results.

============================================================
SAFETY AGAINST HALLUCINATION
============================================================

If the user asks for a paper that cannot be found,
DO NOT generate questions and pretend they are the
original paper.

You may offer:

"Main original paper verify nahi kar saka,
lekin main available related papers de sakta hoon."

============================================================
LANGUAGE
============================================================

Respond in the user's language.

If the user speaks Roman Urdu,
respond naturally in Roman Urdu.

If the user speaks English,
respond in English.

Keep answers easy to understand.

============================================================
IMPORTANT
============================================================

You are a SEARCH AND RETRIEVAL AGENT.

Your job is NOT to guess.

Your job is to FIND, VERIFY, ORGANIZE and EXPLAIN.
"""


# ============================================================
# 8. GEMINI MODEL
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)


# ============================================================
# 9. CREATE AGENT
# ============================================================

agent = create_agent(
    model=model,
    tools=[
        live_google_search
    ],
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# 10. RUN AGENT
# ============================================================

def run_agent(user_message: str):

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": user_message
            }
        ]
    })

    messages = result.get(
        "messages",
        []
    )

    for message in reversed(messages):

        content = getattr(
            message,
            "content",
            ""
        )

        if isinstance(content, str):

            if content.strip():

                return content.strip()

    return (
        "Sorry, mujhe is waqt proper response "
        "generate nahi ho saka."
    )


# ============================================================
# 11. MAIN CHAT LOOP
# ============================================================

def main():

    print()
    print("=" * 65)
    print("                 EXAMVAULT AI")
    print("          Past Paper Research Agent")
    print("=" * 65)

    print()
    print("Live Google Search: ON")
    print("Serper Search: ON")
    print("AI Agent: ON")
    print()
    print("Type 'exit' to close.")
    print()

    while True:

        try:

            user_message = input("You: ").strip()

        except KeyboardInterrupt:

            print()
            print("Agent closed.")
            break

        except EOFError:

            print()
            print("Agent closed.")
            break

        if not user_message:
            continue

        if user_message.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print()
            print("ExamVault AI closed.")
            break

        print()
        print("ExamVault AI: Searching / thinking...")
        print()

        try:

            answer = run_agent(
                user_message
            )

            print(answer)
            print()

        except Exception as error:

            print()
            print("❌ Something went wrong.")
            print()

            print(
                f"Error: {error}"
            )

            print()

            print(
                "Tip: API key, internet connection, "
                "model availability or search service "
                "temporarily unavailable ho sakta hai."
            )

            print()


# ============================================================
# 12. START
# ============================================================

if __name__ == "__main__":
    main()