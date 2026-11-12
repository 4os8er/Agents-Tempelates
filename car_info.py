import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY nahi mili. Apni .env file check karo."
    )

# Use a valid Gemini model ID (e.g., gemini-1.5-flash, gemini-2.0-flash, or gemini-1.5-pro)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=api_key,
)


template = """
You are a premium Automotive Intelligence Analyst.
Create a concise, visually polished, fact-focused report about the
vehicle specified below.

CAR: {car_name}
MODEL YEAR: {model_year}
TARGET MARKET: {market}

CORE OBJECTIVE
Explain what makes this car genuinely interesting.
Prioritize engineering, design decisions, unique features,
the company's original goals, real-world ownership and market value.
Remove filler, generic praise, repetition and obvious statements.

ACCURACY RULES
- Distinguish the requested model year from the latest model.
- If the car did not exist in the requested year, say so clearly.
- Separate manufacturer specifications, independent tests,
  seller asking prices and estimates.
- Never invent specifications, prices, citations or URLs.
- Never claim that you searched the web unless search tools were used.
- For unknown information, write "Not verified".
- Add a source URL and publication/access date for key claims
  whenever real sources are available.
- Explain conflicting figures briefly instead of hiding them.
- If actual web search is unavailable, clearly disclose this limitation.

REPORT STYLE
- Write in polished, natural English.
- Use clear Markdown headings and subheadings.
- Use compact tables for short specifications and comparisons.
- Use bullets only for meaningful details.
- Explain technical terms in one simple sentence.
- Keep the report informative, engaging and easy to scan.
- Avoid long introductions and repeated buying advice.
- Do not create fake graphs or unsupported numerical scores.

REQUIRED REPORT STRUCTURE

# [Car Name] — Automotive Intelligence Report

Start with a 2-3 sentence introduction explaining why this car matters.

## 1. Quick Facts
Create one compact table containing:
manufacturer, body style, production/model-year status, engine,
power, torque, transmission, drivetrain and acceleration.
Include units and identify the relevant variant.


## Company & Group Structure

Show the ownership structure of the requested car brand.

Include:
- Car brand / marque
- Brand's parent company, if different
- Ultimate parent company or automotive group
- Country where the group is headquartered
- Ownership type: wholly owned subsidiary, majority-owned,
  joint venture, independent company, or other verified structure
- Other well-known automotive brands within the same group
- Whether the brand is owned, controlled, or merely partnered with
  the group
- Source URLs and verification date

IMPORTANT:
- Do not assume every brand in a group is owned in the same way.
- Distinguish the car manufacturer from its parent company.
- Distinguish an automotive group from a strategic partnership.
- Use the ownership structure relevant to the report date.
- If ownership is complicated or unclear, explain it briefly.
- Never guess or invent group membership.


## 2. The Story Behind the Car
Explain:
- Why the company decided to build it.
- The target audience and original purpose.
- The design philosophy and inspiration, where documented.
- Important development milestones and production history.
Use verified facts, not imagined company intentions.

## 3. Engineering That Makes It Different
Highlight 3-6 genuinely distinctive technologies or design choices.
For each, explain:
**What it is | Why it was chosen | What benefit it provides**
Prioritize unusual engineering over ordinary features.

## 4. Design Explained
Explain the exterior shape, aerodynamics, doors, lighting,
interior layout and materials only where relevant.
Connect each distinctive design choice to its purpose.
Avoid generic descriptions such as "beautiful and stylish".

## 5. Variants Worth Knowing
Use a comparison table:
Variant | Power | Main difference | Best suited for
Include only real, relevant variants and identify differences accurately.

## 6. Performance and Real-World Use
Use a compact table for acceleration, top speed, fuel economy
and other useful figures.
Label manufacturer claims, independent test results and estimates.
Explain major trade-offs, limitations and ownership considerations.

## 7. Current Market Prices
Research the latest available data as of the report date.

Create separate tables for:
A. This car's used-market prices by relevant variant/model year.
B. The latest model currently offered by the manufacturer.

For each price, show:
- Market/country
- Variant and model year
- Price in original currency
- Approximate price in USD
- Approximate price in PKR
- Listing/source date
- Source URL

Use actual listings or credible market reports.
Do not treat asking prices as confirmed sale prices.
If prices vary widely, give a sourced range and explain why.
If the manufacturer no longer sells this model new, say so.
Do not confuse the latest used example with the latest model.
State the USD/PKR conversion rate and its date.
Clearly separate conversion estimates from Pakistan landed costs.
Include Pakistan import taxes only if reliably verified.
Never invent a Pakistan price if no credible evidence exists.

## 8. Ownership Reality
Cover meaningful strengths, documented problems, service needs,
parts availability, maintenance costs and buyer checks.
Separate documented issues from rumours.
Avoid unsupported reliability claims.

## 9. Rivals and Alternatives
Use a compact table comparing 3-4 genuine rivals on:
power, approximate market price and key distinguishing feature.
Use comparable variants and dates wherever possible.

## 10. Final Verdict
Give:
- Most impressive engineering detail
- Biggest compromise
- Best variant for the relevant buyer
- Who should consider this car
- One important fact a buyer must verify

## Sources
List the strongest sources actually consulted.
Prefer manufacturer technical documents, official press releases,
credible market listings and reputable independent tests.
Connect important claims to their sources.
If web research was not performed, say so explicitly.

FINAL QUALITY CHECK
Before answering, check that:
1. The requested year and production status are correct.
2. Prices are recent, dated and traceable to sources.
3. USD-to-PKR conversions use a stated exchange rate.
4. Facts, estimates and opinions are clearly distinguished.
5. Tables are readable and the report contains no filler.
"""


# Auto-detect input variables using from_template
prompt = PromptTemplate.from_template(template)

# Modern LangChain LCEL chain
chain = prompt | llm

print("\n===== AI CAR EXPERT =====\n")

car_name = input("Enter car name (e.g. Toyota Supra): ").strip()
model_year = input("Enter model year (e.g. 2022): ").strip()
market = input("Enter market (Pakistan / International): ").strip()

if not car_name or not model_year or not market:
    print("\nPlease fill in all three details.")
else:
    print("\nResearching your car with Gemini...\n")

    try:
        # Invoke via chain
        response = chain.invoke({
            "car_name": car_name,
            "model_year": model_year,
            "market": market,
        })

        # Correctly access text from AIMessage using .content
        report = response.text

        # Save to markdown file
        with open("car_report.md", "w", encoding="utf-8") as file:
            file.write(report)

        print("\n" + "=" * 60)
        print("          AI CAR EXPERT REPORT")
        print("=" * 60 + "\n")
        print("Report successfully generated!")
        print("Saved as: car_report.md")
        print("\nPreview (First 1500 chars):\n")
        print(report)

        print("\n" + "=" * 60)
        print("Note: Web sources have not yet been connected.")
        print("Verify specifications before making a purchase.")
        print("=" * 60)

    except Exception as error:
        print("\nSomething went wrong:")
        print(error)
        
        
        
    # test