import os
import json
from datetime import datetime
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

# Load .env
load_dotenv()

API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not API_KEY:
    raise ValueError("Missing FIRECRAWL_API_KEY in .env")

print(API_KEY)

# page to crawl
BASE_URL = "https://courses.cs.washington.edu/courses/cse121/26wi/"

PAGES_TO_VISIT = [
    "",
    "syllabus/", 
    "assignments/",
    "resubs/",
    "exams/",
    "getting_help/",
    "staff/",
    "rubrics/",
    "resources/",
    "course_tools/"
]

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"{OUTPUT_DIR}/cse121_dump_{timestamp}.json"

# Start crawling
firecrawl = FirecrawlApp(api_key=API_KEY)

# Scrape a website:
all_results = []

for page in PAGES_TO_VISIT:
    url = BASE_URL + page
    print(f"Scraping {url}")
    result = firecrawl.scrape(url, formats={"markdown": True, "metadata": True})
    print("Result: ", result)
    result_dict = result.model_dump()
    all_results.append({
        "url": url,
        "content": result_dict
    })


# Save result
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)


