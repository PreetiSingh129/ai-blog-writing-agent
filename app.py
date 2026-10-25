from dotenv import load_dotenv
import os
from typing import TypedDict
from langgraph.graph import StateGraph,END
from langchain_google_genai import ChatGoogleGenerativeAI

class BlogState(TypedDict):
    topic: str
    blog: str
    review: str
    retry_count: int

load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")





llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    timeout=30,
    max_retries=0
)
review_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    timeout=30,
    max_retries=0
)


def write_blog(state: BlogState):

    topic = state["topic"]

    prompt = f"""
    Create a blog on the topic: {topic}

    First think of a suitable outline internally, then write the complete blog.

    Requirements:
    - Use simple language
    - Add clear headings
    - Explain each point clearly
    - Include introduction and conclusion
    - Make it useful for beginners
    """
    print("🟡 Sending request to Gemini...")
    response = llm.invoke(prompt)
    print("🟢 Gemini response received!")
    return {
        "blog": get_text(response.content)
    }

def review_blog(state: BlogState):

    blog = state["blog"]

    prompt = f"""
    Review this blog:

    {blog}

    Check clarity, structure, simple language and missing important points.

    Start your response with exactly one of:
    GOOD
    NEEDS_IMPROVEMENT

    Then give only a short reason.
    """
    print("🟡 Sending review request to Gemini...")
    response = review_llm.invoke(prompt)
    print("🟢 Review response received!")
    return {
        "review": get_text(response.content)
    }

def get_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    return str(content)
def review_decision(state: BlogState):

    review = (state["review"]).upper()
    retry_count = state["retry_count"]
    if review.strip().startswith("GOOD"):
        return "good"
    if retry_count >= 2:
        return "good"

    return "rewrite"

def rewrite_blog(state: BlogState):

    topic = state["topic"]
    blog = state["blog"]
    review = state["review"]
    retry_count = state["retry_count"]
    prompt = f"""
    Rewrite and improve the following blog.

    Topic:
    {topic}

    Current Blog:
    {blog}

    Review:
    {review}

    Requirements:
    - Fix the problems mentioned in the review
    - Use simple and clear language
    - Keep a good structure with headings
    - Do not remove useful information
    - Make the blog engaging for beginners
    """
    print("🟡 Sending rewrite request to Gemini...")
    response = llm.invoke(prompt)
    print("🟢 Gemini response received!")

    return {
        "blog": get_text(response.content),
        "retry_count": retry_count + 1
    }
graph = StateGraph(BlogState)


graph.set_entry_point("write_blog")

graph.add_node("write_blog", write_blog)
graph.add_node("review_blog", review_blog)
graph.add_node("rewrite_blog", rewrite_blog)

graph.add_edge("write_blog", "review_blog")

graph.add_conditional_edges(
    "review_blog",
    review_decision,
    {
        "good": END,
        "rewrite": "rewrite_blog"
    }
)

graph.add_edge("rewrite_blog", "review_blog")

app = graph.compile()

result = app.invoke({
    "topic": "Generative AI",
   
    "blog": "",
    "review": "",
    "retry_count": 0
})

print("\nBLOG:")
print(result["blog"])
print("\nREVIEW:")
print(result["review"])