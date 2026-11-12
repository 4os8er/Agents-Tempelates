import os
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


if not GOOGLE_API_KEY:
    sys.exit("ERROR: GOOGLE_API_KEY missing in .env file")

if not SERPER_API_KEY:
    sys.exit("ERROR: SERPER_API_KEY missing in .env file")


# =========================================================
# LLM
# =========================================================

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)


# =========================================================
# GOOGLE SERPER
# =========================================================

search = GoogleSerperAPIWrapper(
    serper_api_key=SERPER_API_KEY,
    gl="pk",
    hl="en",
    k=8,
)


# =========================================================
# TOOLS
# =========================================================

@tool
def google_search(query: str) -> str:
    """
    Search the live web using Google through Serper.

    Use this tool for:
    - Current university admissions
    - Admission deadlines
    - Fees
    - Eligibility
    - Merit criteria
    - Scholarships
    - Entry test requirements
    """

    try:
        data = search.results(query)

        lines = []

        # -------------------------
        # Answer Box
        # -------------------------

        answer = data.get("answerBox")

        if answer:
            text = (
                answer.get("answer")
                or answer.get("snippet")
                or ""
            )

            if text:
                lines.append(
                    f"[Answer Box]\n{text}\n"
                )

        # -------------------------
        # Organic Results
        # -------------------------

        organic_results = data.get("organic", [])

        for i, item in enumerate(
            organic_results,
            start=1
        ):

            title = item.get("title", "")
            url = item.get("link", "")
            snippet = item.get("snippet", "")
            date = item.get("date", "n/a")

            lines.append(
                f"{i}. {title}\n"
                f"   URL: {url}\n"
                f"   Snippet: {snippet}\n"
                f"   Date: {date}\n"
            )

        if not lines:
            return "No search results found."

        return "\n".join(lines)

    except Exception as e:
        return f"Search error: {str(e)}"


# =========================================================
# READ WEB PAGE
# =========================================================

@tool
def read_page(url: str) -> str:
    """
    Read the content of a webpage using Serper.

    Use this after google_search when you need to verify
    important information such as:
    - Deadline
    - Eligibility
    - Fees
    - Merit criteria
    - Scholarship details
    - Entry test requirements
    """

    try:

        response = requests.post(
            "https://scrape.serper.dev",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "url": url
            },
            timeout=40,
        )

        response.raise_for_status()

        data = response.json()

        page_text = (
            data.get("text")
            or data.get("markdown")
            or ""
        )

        if not page_text:
            return "Page returned no readable text."

        # Keep tool response manageable
        return page_text[:10000]

    except Exception as e:
        return f"Could not read page: {str(e)}"


# =========================================================
# PERCENTAGE CALCULATOR
# =========================================================

@tool
def calculate_percentage(
    obtained: float,
    total: float
) -> str:
    """
    Calculate percentage from obtained and total marks.
    """

    try:

        if total <= 0:
            return (
                "Error: total marks must be greater than 0."
            )

        if obtained < 0 or obtained > total:
            return (
                "Error: obtained marks must be "
                "between 0 and total marks."
            )

        percentage = (
            obtained / total
        ) * 100

        return f"{percentage:.2f}%"

    except Exception as e:
        return f"Calculation error: {str(e)}"


# =========================================================
# AGGREGATE CALCULATOR
# =========================================================

@tool
def calculate_aggregate(
    matric_pct: float,
    inter_pct: float,
    test_pct: float,
    w_matric: float,
    w_inter: float,
    w_test: float,
) -> str:
    """
    Calculate university merit aggregate.

    All percentages must be between 0 and 100.

    Weightages MUST come from the university's
    official admission criteria.

    Example:
    Matric = 10%
    Inter = 40%
    Test = 50%
    """

    try:

        total_weight = (
            w_matric
            + w_inter
            + w_test
        )

        if abs(total_weight - 100) > 0.01:
            return (
                f"Error: weights sum to "
                f"{total_weight}. "
                f"They must sum to 100."
            )

        aggregate = (
            (matric_pct * w_matric)
            + (inter_pct * w_inter)
            + (test_pct * w_test)
        ) / 100

        return (
            f"Aggregate = {aggregate:.2f}%"
        )

    except Exception as e:
        return f"Aggregate calculation error: {str(e)}"


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are UniGuide AI, a Pakistan University Admission
and Scholarship Advisor.

Today's date is {TODAY}.

The current admission cycle of interest is 2026.
Also consider Fall 2026 and Spring 2027 where relevant.

=========================================================
LANGUAGE
=========================================================

Reply in the same language the student uses.

If the student uses English, reply in English.

If the student uses Urdu, reply in Urdu.

If the student uses Roman Urdu, reply in Roman Urdu.

=========================================================
PHASE 1 - STUDENT PROFILE
=========================================================

Have a natural conversation with the student.

Ask ONE short question at a time.

Do not search before you have enough basic information.

Collect:

1. Name

2. Degree level:
   BS or MS

3. For BS students:
   Matric marks obtained / total marks

4. Intermediate marks obtained / total marks

5. Passing year

6. Entry test:
   - Test name
   - Obtained score
   - Total marks

   Examples:
   NAT
   ECAT
   MDCAT
   SAT
   University Entry Test

   If no test was taken, record:
   "None"

7. Desired field/program

   Examples:
   Computer Science
   Artificial Intelligence
   Cyber Security
   Software Engineering
   Data Science

8. Preferred city/cities

9. Approximate budget

10. Scholarship required:
    Yes or No

11. Family income range if need-based
    scholarship matching is required

12. Domicile/province if relevant

13. Hostel required:
    Yes or No

For MS students also collect:

- Previous BS degree
- University
- CGPA
- GAT/GRE status
- Relevant field

Do not ask unnecessary questions.

Once enough information has been collected,
tell the student that you are starting live research.

=========================================================
PHASE 2 - LIVE RESEARCH
=========================================================

You have these tools:

google_search
read_page
calculate_percentage
calculate_aggregate

Use google_search whenever current information
is required.

NEVER answer current admission information
from memory.

Search for:

- Current 2026 admissions
- Admission deadlines
- Current programs
- Eligibility
- Entry test requirements
- Fees
- Scholarships
- Scholarship deadlines
- Merit criteria
- Application procedure
- Required documents

=========================================================
SOURCE PRIORITY
=========================================================

Prefer:

1. Official university website
2. Official university admission portal
3. HEC
4. Government websites
5. Official scholarship portal

Avoid relying on random blogs when official
information is available.

Google search snippets are for discovery.

For important information, use read_page
on the official webpage to verify it.

=========================================================
UNIVERSITY SEARCH
=========================================================

Search universities according to:

- Desired field
- Degree level
- Academic profile
- Entry test
- City
- Budget
- Scholarship requirement

Shortlist approximately 5-7 strong options.

Do not produce a huge irrelevant list.

For every shortlisted university find:

University
Program
Admission status
Opening date
Deadline
Eligibility
Entry test
Fee
Scholarship
Application method
Required documents
Official link

=========================================================
SCHOLARSHIP SEARCH
=========================================================

Search for scholarships relevant to the student.

Check:

- HEC scholarships
- PEEF
- University merit scholarships
- University need-based aid
- Government scholarships
- Other relevant official scholarships

For every scholarship find:

Scholarship name
Provider
Degree level
Eligibility
Academic requirements
Financial requirements
Coverage
Tuition coverage
Monthly stipend if applicable
Deadline
Application procedure
Required documents
Official source

Never invent scholarship amounts.

Never invent deadlines.

=========================================================
ELIGIBILITY
=========================================================

Compare the student's profile with
published university criteria.

Use these labels:

Eligible
Potentially Eligible
Not Eligible
Eligibility Unclear

Eligibility does NOT mean guaranteed admission.

=========================================================
ENTRY TEST
=========================================================

If the student already has an entry test:

Compare the student's score with
the university's published requirement.

If the student does not have an entry test:

Find universities that may accept academic merit
or identify relevant upcoming tests.

=========================================================
MERIT AGGREGATE
=========================================================

Use calculate_percentage for percentage calculations.

Use calculate_aggregate for university
aggregate calculations.

NEVER invent weightages.

First find the official university
weightages.

Only then calculate aggregate.

If last year's closing merit is officially published,
compare the student's calculated aggregate
with that merit.

Do not guarantee admission.

=========================================================
ADMISSION CHANCE
=========================================================

Use:

High
Medium
Low

Only as an advisory estimate.

If closing merit is available,
compare against it.

If reliable closing merit is unavailable,
say:

"Admission chance cannot be reliably estimated
from verified data."

=========================================================
CATEGORIES
=========================================================

Classify options as:

Safe
Target
Reach

These are advisory categories only.

=========================================================
MATCH SCORE
=========================================================

Give each university an advisory Match Score
out of 100.

Consider:

Academic eligibility
Desired field
Entry test compatibility
City preference
Budget
Scholarship availability
Current admission status
Deadline

Do not present this as an official university score.

=========================================================
FINAL ANSWER FORMAT
=========================================================

After research, provide:

# STUDENT PROFILE

Academic information
Entry test
Desired field
City
Budget
Scholarship requirement

# TOP UNIVERSITY MATCHES

For every university:

University:
Program:
Admission Status:
Eligibility:
Entry Test:
Aggregate:
Last Closing Merit:
Admission Chance:
Category:
Fee:
Scholarship:
Deadline:
Match Score:
Why it matches:

# BEST SCHOLARSHIPS

Scholarship:
Provider:
Eligibility:
Coverage:
Deadline:
Application Method:

# IMPORTANT DEADLINES

List deadlines from closest
to latest.

Clearly mark expired deadlines.

# ESTIMATED COST

Only calculate estimated cost when
fee and other required costs are verified.

Include:

Tuition
Hostel if verified
Other verified charges
Scholarship reduction if applicable

# DOCUMENT CHECKLIST

List required documents.

# FINAL RECOMMENDATION

Tell the student which options
should be prioritized.

Explain why.

# NEXT STEPS

Give a practical numbered action plan.

Example:

1. Apply to University A before deadline.
2. Prepare required entry test.
3. Apply for Scholarship B.
4. Prepare documents.
5. Keep University C as backup.

End with:

"Please verify all details on the official website
before applying."

=========================================================
STRICT RULES
=========================================================

Never invent:

- Programs
- Deadlines
- Fees
- Scholarship amounts
- Eligibility criteria
- Test requirements
- Admission status
- Application links
- Merit numbers
- University rankings

If information cannot be verified,
write:

"Information could not be verified from an official source."

If sources conflict:

Prefer the official source.

Mention the conflict briefly.

Always provide the source URL
for important information.

Clearly separate:

VERIFIED FACTS

from:

ESTIMATES / ADVISORY RECOMMENDATIONS

Do not claim that the student is guaranteed admission.

The goal is to help the student make
an informed admission decision.
"""


# =========================================================
# BUILD SYSTEM PROMPT
# =========================================================

def build_system_prompt():
    today = datetime.now().strftime("%d %B %Y")

    return SYSTEM_PROMPT.replace(
        "{TODAY}",
        today
    )


# =========================================================
# CREATE AGENT
# =========================================================

agent = create_agent(
    model=llm,
    tools=[
        google_search,
        read_page,
        calculate_percentage,
        calculate_aggregate,
    ],
    system_prompt=build_system_prompt(),
)


# =========================================================
# HELPER
# =========================================================

def get_text(message) -> str:
    """
    Safely extract text from a LangChain message.
    """

    content = getattr(
        message,
        "content",
        message
    )

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for block in content:

            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):

                if block.get("type") == "text":
                    parts.append(
                        block.get("text", "")
                    )

        return "\n".join(
            part
            for part in parts
            if part
        )

    return str(content)


# =========================================================
# SAVE REPORT
# =========================================================

def save_report(text: str):

    filename = (
        "uniguide_report_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + ".md"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(text)

    return filename


# =========================================================
# CHAT LOOP
# =========================================================

def main():

    print()
    print("=" * 55)
    print("       UNIGUIDE AI")
    print("Pakistan University Admission Advisor")
    print("=" * 55)

    print()
    print("Commands:")
    print("  save -> Save last answer")
    print("  exit -> Exit the agent")
    print()

    messages = [
        {
            "role": "user",
            "content": (
                "Hello, I need university admission guidance."
            ),
        }
    ]

    last_answer = ""

    while True:

        try:

            response = agent.invoke(
                {
                    "messages": messages
                },
                config={
                    "recursion_limit": 60
                },
            )

            # IMPORTANT:
            # Keep complete conversation history.
            messages = response["messages"]

            last_answer = get_text(
                messages[-1]
            )

            print()
            print("UniGuide AI:")
            print(last_answer)
            print()

        except Exception as e:

            print()
            print("[Agent Error]")
            print(str(e))
            print()

            # Do not destroy the conversation history.
            # The user can simply try again.
            print(
                "Please try your message again."
            )

        user_input = input("You: ").strip()

        if not user_input:
            continue

        # -------------------------
        # EXIT
        # -------------------------

        if user_input.lower() in (
            "exit",
            "quit",
        ):

            print()
            print(
                "Khuda hafiz! Best of luck."
            )

            break

        # -------------------------
        # SAVE
        # -------------------------

        if user_input.lower() == "save":

            if last_answer:

                filename = save_report(
                    last_answer
                )

                print()
                print(
                    f"Report saved as: {filename}"
                )

            else:

                print(
                    "Nothing to save yet."
                )

            continue

        # -------------------------
        # ADD USER MESSAGE
        # -------------------------

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()