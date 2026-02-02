from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("FIRECRAWL_API_KEY")

print(API_KEY)

