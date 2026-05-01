from dotenv import load_dotenv
from claude_client import claude_complete

load_dotenv()


def generate_solution(review, category, count):

    prompt = f"""
You are a hotel quality management expert.

Analyze this complaint:

Category: {category}
Review: {review}
Monthly occurrences: {count}

Give:

1. Root Cause
2. Immediate Fix
3. Long Term Fix
4. Priority (Low/Medium/High)
5. Management Advice
"""

    return claude_complete(
        prompt.strip(),
        max_tokens=1200,
        system="You are a hospitality expert.",
    )