"""
ExamVault AI - Past Paper Research Agent

Finds REAL past papers on the live web using Google results (via Serper)
and a Gemini model (via LangChain).

Setup:
    pip install -U langchain langchain-google-genai requests python-dotenv

    Create a .env file next to this script:
        GOOGLE_API_KEY=your_gemini_key
        SERPER_API_KEY=your_serper_key
        # optional:
        # GEMINI_MODEL=gemini-3.5-flash-lite

Run:
    python examvault.py
"""

import json
import os
import re
import sys
import time
from datetime import date
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import GraphRecursionError


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Change the model without editing code:  GEMINI_MODEL=gemini-3.5-flash
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Serper's real endpoint (POST, key sent in the X-API-KEY header).
SERPER_URL = "https://google.serper.dev/search"

MAX_SEARCH_RESULTS = 10          # results requested from Serper per search
MAX_RESULTS_TO_AGENT = 8         # top-ranked results handed back to the model
SEARCH_TIMEOUT = 30              # seconds
MAX_SEARCH_ATTEMPTS = 3          # retries for timeouts / 429 / 5xx only
RECURSION_LIMIT = 24             # hard cap on agent steps per user message
MAX_HISTORY_TURNS = 6            # how many past user turns the agent remembers

FALLBACK_ANSWER = (
    "Sorry, mujhe is waqt proper response generate nahi ho saka."
)


def check_environment() -> bool:
    """Return True if all required keys exist, else print what's missing."""
    ok = True

    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY is missing from .env")
        ok = False

    if not SERPER_API_KEY:
        print("❌ SERPER_API_KEY is missing from .env")
        ok = False

    return ok


# ============================================================
# 2. SEARCH FUNCTION
# ============================================================

def parse_serper_results(data: dict) -> list:
    """Convert Serper's JSON into a clean list of result dicts.

    Serper returns organic hits as: title, link, snippet, position, date.
    (The field is "link", not "url", and there is no "domain" field.)
    """

    clean_results = []

    for index, item in enumerate(data.get("organic", []) or [], start=1):

        link = item.get("link", "")

        if not link:
            continue

        domain = urlparse(link).netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        clean_results.append({
            "position": item.get("position", index),
            "title": item.get("title", ""),
            "url": link,
            "domain": domain,
            "snippet": item.get("snippet", ""),
            "date": item.get("date", ""),
        })

    return clean_results


def serper_search(query: str, num: int = MAX_SEARCH_RESULTS) -> list:
    """Run one live Google search through Serper."""

    if not query or not query.strip():
        return []

    headers = {
        "X-API-KEY": SERPER_API_KEY or "",
        "Content-Type": "application/json",
    }

    payload = {
        "q": query.strip(),
        "gl": "pk",      # country: Pakistan
        "hl": "en",      # interface language
        "num": num,
    }

    last_error = "unknown error"

    for attempt in range(1, MAX_SEARCH_ATTEMPTS + 1):

        try:
            response = requests.post(
                SERPER_URL,
                headers=headers,
                json=payload,
                timeout=SEARCH_TIMEOUT,
            )

        except requests.exceptions.Timeout:
            last_error = "Search request timed out."

        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {e}"

        else:

            if response.status_code == 200:

                try:
                    data = response.json()
                except ValueError:
                    last_error = "Invalid JSON returned by Serper."
                else:
                    return parse_serper_results(data)

            elif response.status_code in (400, 401, 403):
                # Bad key / no credits / bad request: retrying won't help.
                raise RuntimeError(
                    f"Serper rejected the request "
                    f"(HTTP {response.status_code}). Check SERPER_API_KEY "
                    f"and your Serper credits. "
                    f"Details: {response.text[:300]}"
                )

            else:
                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )

        if attempt < MAX_SEARCH_ATTEMPTS:
            time.sleep(attempt)

    raise RuntimeError(
        f"Google search failed after {MAX_SEARCH_ATTEMPTS} attempts. "
        f"Reason: {last_error}"
    )


# ============================================================
# 3. TEXT UTILITIES
# ============================================================

def normalize(text) -> str:
    """Lowercase and turn every run of punctuation into one space.

    "NAT-ICS_2025.pdf" -> "nat ics 2025 pdf", so "NAT-ICS", "NAT ICS"
    and "nat_ics" all compare equal.
    """

    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)

    return text.strip()


def contains(text: str, value: str) -> bool:
    """Whole-word match, so 'ics' does not match inside 'physics'."""

    value_n = normalize(value)

    if not value_n:
        return False

    return f" {value_n} " in f" {normalize(text)} "


def extract_years(value) -> list:
    """'2025' -> [2025];  '2020-2025' or '2020 to 2025' -> [2020..2025]."""

    if not value:
        return []

    text = str(value)

    years = sorted({
        int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    })

    is_range = re.search(r"\d\s*(?:-|–|—|to|till|until)\s*\d", text, re.I)

    if len(years) >= 2 and is_range:
        low, high = years[0], years[-1]
        if 0 < high - low <= 30:
            return list(range(low, high + 1))

    return years


def year_matches(text: str, year_value) -> bool:

    years = extract_years(year_value)

    if years:
        return any(contains(text, str(y)) for y in years)

    return contains(text, year_value)


OFFICIAL_DOMAIN_HINTS = (".edu.pk", ".gov.pk", "nts.org.pk")


# ============================================================
# 4. RESULT SCORING
# ============================================================

def score_result(result: dict, requirements: dict):
    """Heuristic relevance score based on title/snippet/domain/url text.

    This measures how well the *search result text* matches the request.
    It does NOT prove the linked file really is that paper.
    """

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

    if year and year_matches(combined_text, year):
        score += 20
        reasons.append("year matched")

    if contains(combined_text, shift):
        score += 10
        reasons.append("shift matched")

    # Check the RAW url: normalize() strips the dot, so ".pdf" would
    # never be found in a normalized string.
    if ".pdf" in result.get("url", "").lower():
        score += 10
        reasons.append("PDF")

    domain = result.get("domain", "").lower()

    if any(domain.endswith(hint) for hint in OFFICIAL_DOMAIN_HINTS):
        score += 10
        reasons.append("official-looking domain")

    return score, reasons


# ============================================================
# 5. RESULT RANKING
# ============================================================

def rank_results(results: list, requirements: dict) -> list:

    ranked = []

    for result in results:

        score, reasons = score_result(result, requirements)

        ranked.append({
            "score": score,
            "reasons": reasons,
            "result": result,
        })

    # Highest score first; ties keep Google's original order.
    ranked.sort(
        key=lambda x: (-x["score"], x["result"].get("position", 0))
    )

    return ranked


# ============================================================
# 6. SEARCH TOOL FOR LANGCHAIN
# ============================================================

@tool
def live_google_search(
    query: str,
    exam: str = "",
    category: str = "",
    university: str = "",
    board: str = "",
    program: str = "",
    year: str = "",
    shift: str = "",
) -> str:
    """
    Search Google live (via Serper) for real past papers, answer keys,
    university entry tests, NTS/NAT papers, board papers and related
    educational documents. Never invent URLs or papers - only use what
    this tool returns.

    Args:
        query: The Google search query, e.g. "NAT ICS 2025 past paper PDF".
        exam: Exam name if known, e.g. "NAT", "MDCAT", "ECAT".
        category: Category if known, e.g. "NAT-ICS".
        university: University if known, e.g. "NUST", "FAST".
        board: Board if known, e.g. "BISE Lahore".
        program: Program if known, e.g. "BSCS".
        year: Year the user wants, e.g. "2025" or "2020-2025".
        shift: Shift if known, e.g. "morning".

    Fill the optional fields from what the user asked: results are
    re-ranked by how well their title/snippet/URL match those fields.
    """

    requirements = {
        key: value.strip()
        for key, value in {
            "exam": exam,
            "category": category,
            "university": university,
            "board": board,
            "program": program,
            "year": year,
            "shift": shift,
        }.items()
        if value and value.strip()
    }

    try:
        results = serper_search(query)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "query": query,
            "error": str(e),
        }, ensure_ascii=False)

    if not results:
        return json.dumps({
            "status": "no_results",
            "query": query,
            "results": [],
        }, ensure_ascii=False)

    ranked = rank_results(results, requirements)[:MAX_RESULTS_TO_AGENT]

    return json.dumps({
        "status": "success",
        "query": query,
        "requirements_used": requirements,
        "note": (
            "relevance_score is a heuristic from title/snippet/url text "
            "only. It does not prove the file is the exact paper."
        ),
        "results": [
            {
                **item["result"],
                "relevance_score": item["score"],
                "match_reasons": item["reasons"],
            }
            for item in ranked
        ],
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

14. Never enter an infinite search loop. Make at most
    6 searches for one user request.

15. Do not perform unnecessary searches.

16. Do not ask for shift if that exam normally
    does not have a meaningful shift distinction.

============================================================
HOW TO USE THE SEARCH TOOL
============================================================

The tool is live_google_search.

- Put the Google query in "query".
- ALSO fill the optional fields (exam, category, university,
  board, program, year, shift) with whatever the user has
  told you. The tool uses them to rank results.
- Every result has a relevance_score and match_reasons.
  These come only from the title, snippet and URL text.
  A high score is NOT proof that the exact paper exists.
- If the status is "error", tell the user the search failed
  and why. Do not pretend you found results.
- If the status is "no_results", try ONE differently worded
  query, then be honest if still nothing is found.
- For a multi-year request, search one year at a time
  (or the most efficient grouping), then organize by year.

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


def build_system_prompt() -> str:
    """Add today's date so the model knows what 'latest' means."""

    return (
        f"{SYSTEM_PROMPT}\n"
        f"TODAY'S DATE: {date.today().isoformat()}\n"
    )


# ============================================================
# 8. GEMINI MODEL + 9. CREATE AGENT
# ============================================================

def build_agent():
    """Create the LangChain agent (done lazily so imports never need keys).

    Note: temperature is left at Gemini's default on purpose. Google
    recommends keeping it at 1.0 for Gemini 3 models; lower values can
    cause looping or degraded results.
    """

    model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        max_retries=2,
    )

    return create_agent(
        model=model,
        tools=[live_google_search],
        system_prompt=build_system_prompt(),
    )


_agent = None


def get_agent():

    global _agent

    if _agent is None:
        _agent = build_agent()

    return _agent


# ============================================================
# 10. RUN AGENT (WITH CONVERSATION MEMORY)
# ============================================================

def message_text(content) -> str:
    """Pull plain text out of a message's content.

    Gemini 3 can return content as a LIST of blocks (text, thinking,
    signatures...) instead of a plain string.
    """

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        parts = []

        for block in content:

            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))

        return "\n".join(p for p in parts if p).strip()

    return ""


def extract_answer(messages: list) -> str:
    """Return the last AI message that actually has text for the user."""

    for message in reversed(messages):

        if isinstance(message, AIMessage) and not message.tool_calls:

            text = message_text(message.content)

            if text:
                return text

    return FALLBACK_ANSWER


def trim_history(messages: list, max_turns: int = MAX_HISTORY_TURNS) -> list:
    """Keep only the last few user turns.

    Cuts only at a user message, so tool calls are never separated from
    their tool results (which would make the API reject the request).
    """

    starts = [
        i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
    ]

    if len(starts) <= max_turns:
        return messages

    return messages[starts[-max_turns]:]


def run_agent(user_message: str, history: list = None):
    """Send one user message. Returns (answer_text, updated_history).

    The full message objects are stored as-is (not just text) so the
    model's reasoning signatures survive between turns.
    """

    history = list(history or [])

    messages = history + [HumanMessage(content=user_message)]

    try:
        result = get_agent().invoke(
            {"messages": messages},
            config={"recursion_limit": RECURSION_LIMIT},
        )

    except GraphRecursionError:
        return (
            "Search bohat lambi ho gayi aur main roki gayi. "
            "Please apni request thori aur specific kar ke dobara "
            "bhejein (exam, category, year).",
            history,
        )

    new_messages = result.get("messages", [])

    return extract_answer(new_messages), trim_history(new_messages)


# ============================================================
# 11. MAIN CHAT LOOP
# ============================================================

def main():

    # Emoji / Urdu text can crash some Windows consoles; force UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    if not check_environment():
        sys.exit(1)

    print()
    print("=" * 65)
    print("                 EXAMVAULT AI")
    print("          Past Paper Research Agent")
    print("=" * 65)

    print()
    print("Live Google Search: ON")
    print("Serper Search: ON")
    print(f"AI Agent: ON ({GEMINI_MODEL})")
    print()
    print("Type 'new' to start a fresh search, 'exit' to close.")
    print()

    history = []

    while True:

        try:
            user_message = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print()
            print("Agent closed.")
            break

        if not user_message:
            continue

        command = user_message.lower()

        if command in ("exit", "quit", "q"):
            print()
            print("ExamVault AI closed.")
            break

        if command in ("new", "reset", "clear"):
            history = []
            print()
            print("ExamVault AI: Naya search shuru. Batayein kya chahiye?")
            print()
            continue

        print()
        print("ExamVault AI: Searching / thinking...")
        print()

        try:
            answer, history = run_agent(user_message, history)

            print(answer)
            print()

        except Exception as error:

            print()
            print("❌ Something went wrong.")
            print()
            print(f"Error: {error}")
            print()
            print(
                "Tip: API key, internet connection, model availability "
                f"('{GEMINI_MODEL}') or search service "
                "temporarily unavailable ho sakta hai. You can switch "
                "models with GEMINI_MODEL in .env."
            )
            print()


# ============================================================
# 12. START
# ============================================================

if __name__ == "__main__":
    main()