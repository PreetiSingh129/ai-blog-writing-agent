import os
import time
from datetime import datetime
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Blog Studio",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .app-sub {
        opacity: .65;
        font-size: .95rem;
        margin: .25rem 0 1.25rem 0;
    }

    .section-lbl {
        font-size: .72rem;
        font-weight: 600;
        letter-spacing: .08em;
        text-transform: uppercase;
        opacity: .55;
        margin: 1.2rem 0 .5rem 0;
    }

    /* Pipeline */
    .pipe {
        display: flex;
        align-items: flex-start;
        padding: 18px 8px;
    }

    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        min-width: 110px;
        text-align: center;
    }

    .dot {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: .85rem;
        border: 2px solid rgba(128,128,128,.4);
        opacity: .7;
    }

    .lbl {
        margin-top: 8px;
        font-weight: 600;
        font-size: .85rem;
    }

    .sub {
        font-size: .72rem;
        opacity: .6;
        margin-top: 2px;
        min-height: 1em;
    }

    .bar {
        flex: 1;
        height: 2px;
        background: rgba(128,128,128,.35);
        margin-top: 17px;
    }

    .step.done .dot {
        background: #16a34a;
        border-color: #16a34a;
        color: #fff;
        opacity: 1;
    }

    .step.active .dot {
        background: #2563eb;
        border-color: #2563eb;
        color: #fff;
        opacity: 1;
        animation: pulse 1.4s infinite;
    }

    .step.warn .dot {
        background: #d97706;
        border-color: #d97706;
        color: #fff;
        opacity: 1;
    }

    .bar.done {
        background: #16a34a;
    }

    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(37,99,235,.5);
        }

        70% {
            box-shadow: 0 0 0 10px rgba(37,99,235,0);
        }

        100% {
            box-shadow: 0 0 0 0 rgba(37,99,235,0);
        }
    }

    /* Status */
    .pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 600;
        letter-spacing: .03em;
    }

    .pill.ok {
        background: rgba(22,163,74,.15);
        color: #16a34a;
    }

    .pill.warn {
        background: rgba(217,119,6,.15);
        color: #d97706;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.22);
        padding: 14px 18px;
        border-radius: 10px;
    }

    div[data-testid="stMetricLabel"] p {
        font-size: .78rem;
        opacity: .65;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #2563eb;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        padding: .55rem 1rem;
    }

    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
    }

    /* Blog */
    .reader {
        line-height: 1.75;
        font-size: 1.02rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class BlogState(TypedDict):
    topic: str
    instructions: str
    audience: str
    tone: str
    length: str
    blog: str
    review: str
    retry_count: int
    max_retries: int


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_text(content) -> str:

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    return str(content)


# ---------------------------------------------------------------------------
# Build LangGraph App
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_app(writer_model: str, reviewer_model: str):

    llm = ChatGoogleGenerativeAI(
        model=writer_model,
        google_api_key=api_key,
        timeout=60,
        max_retries=0
    )

    review_llm = ChatGoogleGenerativeAI(
        model=reviewer_model,
        google_api_key=api_key,
        timeout=60,
        max_retries=0
    )

    # -------------------------------------------------------
    # Write Blog
    # -------------------------------------------------------

    def write_blog(state: BlogState):

        extra = (
            f"- Additional instructions: {state['instructions']}"
            if state["instructions"].strip()
            else ""
        )

        prompt = f"""
        Create a blog on the topic: {state['topic']}

        First think of a suitable outline internally,
        then write the complete blog.

        Requirements:

        - Audience: {state['audience']}
        - Tone: {state['tone']}
        - Length: {state['length']}
        - Use simple language
        - Add clear headings using markdown
        - Explain each point clearly
        - Include an introduction
        - Include a conclusion
        {extra}
        """

        response = llm.invoke(prompt)

        return {
            "blog": get_text(response.content)
        }


    # -------------------------------------------------------
    # Review Blog
    # -------------------------------------------------------

    def review_blog(state: BlogState):

        prompt = f"""
        Review this blog:

        {state['blog']}

        Check:

        - Clarity
        - Structure
        - Simple language
        - Missing important points

        Start your response with exactly one of:

        GOOD

        NEEDS_IMPROVEMENT

        Then give only a short reason.
        """

        response = review_llm.invoke(prompt)

        return {
            "review": get_text(response.content)
        }


    # -------------------------------------------------------
    # Decision
    # -------------------------------------------------------

    def review_decision(state: BlogState):

        review = state["review"].strip().upper()

        if review.startswith("GOOD"):
            return "good"

        if state["retry_count"] >= state["max_retries"]:
            return "good"

        return "rewrite"


    # -------------------------------------------------------
    # Rewrite Blog
    # -------------------------------------------------------

    def rewrite_blog(state: BlogState):

        prompt = f"""
        Rewrite and improve the following blog.

        Topic:
        {state['topic']}

        Audience:
        {state['audience']}

        Tone:
        {state['tone']}

        Length:
        {state['length']}

        Additional instructions:
        {state['instructions']}

        Current Blog:
        {state['blog']}

        Review:
        {state['review']}

        Requirements:

        - Fix the problems mentioned in the review
        - Use simple and clear language
        - Keep a good structure with markdown headings
        - Do not remove useful information
        - Make the blog engaging
        """

        response = llm.invoke(prompt)

        return {
            "blog": get_text(response.content),
            "retry_count": state["retry_count"] + 1
        }


    # -------------------------------------------------------
    # LangGraph
    # -------------------------------------------------------

    graph = StateGraph(BlogState)

    graph.add_node("write_blog", write_blog)
    graph.add_node("review_blog", review_blog)
    graph.add_node("rewrite_blog", rewrite_blog)

    graph.set_entry_point("write_blog")

    graph.add_edge(
        "write_blog",
        "review_blog"
    )

    graph.add_conditional_edges(
        "review_blog",
        review_decision,
        {
            "good": END,
            "rewrite": "rewrite_blog"
        }
    )

    graph.add_edge(
        "rewrite_blog",
        "review_blog"
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

STEP_LABELS = [
    ("draft", "Draft"),
    ("review", "Review"),
    ("revise", "Revise"),
    ("final", "Final")
]


def pipeline_html(state: dict, subs: dict) -> str:

    parts = ['<div class="pipe">']

    for i, (key, label) in enumerate(STEP_LABELS):

        current_state = state.get(key, "idle")

        icon = (
            "✓"
            if current_state == "done"
            else "!"
            if current_state == "warn"
            else str(i + 1)
        )

        parts.append(
            f'<div class="step {current_state}">'
            f'<div class="dot">{icon}</div>'
            f'<div class="lbl">{label}</div>'
            f'<div class="sub">{subs.get(key, "")}</div>'
            f'</div>'
        )

        if i < len(STEP_LABELS) - 1:

            bar_cls = (
                "done"
                if current_state in ("done", "warn")
                else ""
            )

            parts.append(
                f'<div class="bar {bar_cls}"></div>'
            )

    parts.append("</div>")

    return "".join(parts)


def is_good(review: str) -> bool:
    return review.strip().upper().startswith("GOOD")


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    st.markdown("### 📝 Blog Studio")

    st.caption(
        "Automated drafting, review and revision."
    )

    # Models
    st.markdown(
        '<div class="section-lbl">Models</div>',
        unsafe_allow_html=True
    )

    writer_model = st.text_input(
        "Writer model",
        value="gemini-3.6-flash"
    )

    reviewer_model = st.text_input(
        "Reviewer model",
        value="gemini-3.5-flash-lite"
    )

    # Content
    st.markdown(
        '<div class="section-lbl">Content</div>',
        unsafe_allow_html=True
    )

    audience = st.selectbox(
        "Audience",
        [
            "Beginners",
            "Intermediate readers",
            "Professionals"
        ]
    )

    tone = st.selectbox(
        "Tone",
        [
            "Friendly and simple",
            "Professional",
            "Conversational",
            "Inspirational"
        ]
    )

    length = st.select_slider(
        "Length",
        options=[
            "Short (~500 words)",
            "Medium (~900 words)",
            "Long (~1500 words)"
        ],
        value="Medium (~900 words)"
    )

    # Workflow
    st.markdown(
        '<div class="section-lbl">Workflow</div>',
        unsafe_allow_html=True
    )

    max_retries = st.slider(
        "Maximum revisions",
        0,
        4,
        2,
        help="How many times the reviewer can send the draft back for rewriting."
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    '<p class="app-title">Blog Studio</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="app-sub">'
    'Generate a blog, have it independently reviewed, and refine it automatically '
    'until it meets the quality bar.'
    '</p>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

with st.container(border=True):

    topic = st.text_input(
        "Topic",
        placeholder="e.g. Generative AI for absolute beginners"
    )

    instructions = st.text_area(
        "Additional instructions (optional)",
        placeholder="e.g. Include real-world examples and end with a short FAQ.",
        height=80
    )

    generate = st.button(
        "🚀 Generate Blog",
        type="primary",
        use_container_width=True
    )


# ---------------------------------------------------------------------------
# Workflow Tracker
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="section-lbl">Workflow status</div>',
    unsafe_allow_html=True
)

pipe_slot = st.empty()

res = st.session_state.get("result")

if res and not generate:

    pipe_slot.markdown(
        pipeline_html(
            res["pipe_state"],
            res["pipe_subs"]
        ),
        unsafe_allow_html=True
    )

else:

    pipe_slot.markdown(
        pipeline_html(
            {},
            {
                "draft": "Waiting",
                "review": "Waiting",
                "revise": "Waiting",
                "final": "Waiting"
            }
        ),
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Execute Workflow
# ---------------------------------------------------------------------------

if generate:

    if len(topic.strip()) < 3:

        st.warning(
            "Please enter a topic (at least 3 characters)."
        )

    elif not api_key:

        st.error(
            "Gemini API key not found. "
            "Please check your .env file."
        )

    else:

        drafts = []
        reviews = []
        log = []

        pstate = {
            "draft": "active"
        }

        psubs = {
            "draft": "In progress",
            "review": "Waiting",
            "revise": "Disabled" if max_retries == 0 else "Waiting",
            "final": "Waiting"
        }

        pipe_slot.markdown(
            pipeline_html(
                pstate,
                psubs
            ),
            unsafe_allow_html=True
        )

        t_start = time.time()
        t_last = t_start

        try:

            app = build_app(
                writer_model,
                reviewer_model
            )

            init_state = {
                "topic": topic.strip(),
                "instructions": instructions.strip(),
                "audience": audience,
                "tone": tone,
                "length": length,
                "blog": "",
                "review": "",
                "retry_count": 0,
                "max_retries": max_retries
            }

            for chunk in app.stream(
                init_state,
                stream_mode="updates"
            ):

                for node, upd in chunk.items():

                    t_now = time.time()

                    dur = f"{t_now - t_last:.1f}s"

                    t_last = t_now

                    # Draft
                    if node == "write_blog":

                        drafts.append(
                            upd["blog"]
                        )

                        log.append({
                            "Time": now(),
                            "Step": "Draft",
                            "Result": "First draft generated",
                            "Duration": dur
                        })

                        pstate.update(
                            draft="done",
                            review="active"
                        )

                        psubs.update(
                            draft=f"{len(upd['blog'].split())} words",
                            review="Pass 1"
                        )

                    # Review
                    elif node == "review_blog":

                        reviews.append(
                            upd["review"]
                        )

                        revisions = len(drafts) - 1

                        ok = is_good(
                            upd["review"]
                        )

                        log.append({
                            "Time": now(),
                            "Step": "Review",
                            "Result": (
                                "Approved"
                                if ok
                                else "Changes requested"
                            ),
                            "Duration": dur
                        })

                        if ok:

                            pstate.update(
                                review="done",
                                final="done"
                            )

                            psubs.update(
                                review=f"Pass {revisions + 1}",
                                final="Approved"
                            )

                        elif revisions >= max_retries:

                            pstate.update(
                                review="done",
                                final="warn"
                            )

                            psubs.update(
                                review=f"Pass {revisions + 1}",
                                final="Revision limit reached"
                            )

                        else:

                            pstate.update(
                                review="done",
                                revise="active"
                            )

                            psubs.update(
                                review=f"Pass {revisions + 1}",
                                revise=f"Revision {revisions + 1} of {max_retries}"
                            )

                    # Rewrite
                    elif node == "rewrite_blog":

                        drafts.append(
                            upd["blog"]
                        )

                        n = upd["retry_count"]

                        log.append({
                            "Time": now(),
                            "Step": "Revise",
                            "Result": f"Revision {n} completed",
                            "Duration": dur
                        })

                        pstate.update(
                            revise="done",
                            review="active"
                        )

                        psubs.update(
                            revise=f"{n} of {max_retries} used",
                            review=f"Pass {n + 1}"
                        )

                    pipe_slot.markdown(
                        pipeline_html(
                            pstate,
                            psubs
                        ),
                        unsafe_allow_html=True
                    )

            st.session_state.result = {
                "topic": topic.strip(),
                "drafts": drafts,
                "reviews": reviews,
                "log": log,
                "elapsed": time.time() - t_start,
                "approved": is_good(reviews[-1]),
                "pipe_state": pstate,
                "pipe_subs": psubs
            }

            res = st.session_state.result

        except Exception as e:

            st.error(
                f"The workflow failed: {e}"
            )

            pipe_slot.markdown(
                pipeline_html(
                    {
                        **pstate,
                        **{
                            k: "warn"
                            for k, v in pstate.items()
                            if v == "active"
                        }
                    },
                    psubs
                ),
                unsafe_allow_html=True
            )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if res:

    final_blog = res["drafts"][-1]

    final_review = res["reviews"][-1]

    words = len(
        final_blog.split()
    )

    st.markdown(
        '<div class="section-lbl">Result</div>',
        unsafe_allow_html=True
    )

    pill = (
        '<span class="pill ok">APPROVED</span>'
        if res["approved"]
        else
        '<span class="pill warn">REVISION LIMIT REACHED</span>'
    )

    st.markdown(
        f"**{res['topic']}** &nbsp; {pill}",
        unsafe_allow_html=True
    )

    st.write("")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Word count",
        f"{words:,}"
    )

    k2.metric(
        "Reading time",
        f"{max(1, round(words / 200))} min"
    )

    k3.metric(
        "Revisions",
        len(res["drafts"]) - 1
    )

    k4.metric(
        "Total time",
        f"{res['elapsed']:.1f}s"
    )

    st.write("")

    tab_blog, tab_review, tab_hist, tab_log = st.tabs(
        [
            "Blog",
            "Reviewer feedback",
            "Revision history",
            "Execution log"
        ]
    )

    file_stub = (
        res["topic"][:40]
        .strip()
        .replace(" ", "_")
        or "blog"
    )

    # Blog
    with tab_blog:

        with st.container(border=True):

            st.markdown(
                f'<div class="reader">\n\n'
                f'{final_blog}\n\n'
                f'</div>',
                unsafe_allow_html=True
            )

        d1, d2 = st.columns(2)

        d1.download_button(
            "Download Markdown (.md)",
            data=final_blog,
            file_name=f"{file_stub}.md",
            mime="text/markdown",
            use_container_width=True
        )

        d2.download_button(
            "Download Text (.txt)",
            data=final_blog,
            file_name=f"{file_stub}.txt",
            mime="text/plain",
            use_container_width=True
        )

        with st.expander("Raw markdown (copy)"):

            st.code(
                final_blog,
                language="markdown"
            )


    # Review
    with tab_review:

        with st.container(border=True):

            st.markdown(
                final_review
            )


    # Revision History
    with tab_hist:

        for i, draft in enumerate(
            res["drafts"]
        ):

            title = (
                "Initial draft"
                if i == 0
                else f"Revision {i}"
            )

            with st.expander(
                f"{title} · {len(draft.split())} words"
            ):

                st.markdown(
                    draft
                )

                if i < len(res["reviews"]):

                    st.caption(
                        "Reviewer verdict"
                    )

                    st.info(
                        res["reviews"][i]
                    )


    # Execution Log
    with tab_log:

        st.dataframe(
            res["log"],
            use_container_width=True,
            hide_index=True
        )