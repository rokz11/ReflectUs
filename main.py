from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import os
import time

from openai import OpenAI

# =========================
# APP SETUP
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# OPENAI CLIENT
# =========================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = "gpt-5.2"

# =========================
# MODELS
# =========================

class CreateSessionInput(BaseModel):
    creator_name: str
    gender: str | None = None
    language: str | None = "en"

class JoinSessionInput(BaseModel):
    session_id: str
    name: str
    gender: str | None = None

class SaveAnswerInput(BaseModel):
    session_id: str
    role: str
    answer: str

class ReadyInput(BaseModel):
    session_id: str
    role: str

# =========================
# IN-MEMORY STORAGE
# =========================

sessions = {}

# =========================
# MEDIATOR PROMPT (EXACT)
# =========================

MEDIATOR_PROMPT = """You are ReflectUS.

You are a neutral, emotionally intelligent observer of relationships.
Your language must remain human, grounded, and readable — never clinical,
never academic, never performative.

Two partners reflected privately on their relationship.
They did not see each other’s answers.
Individual responses must remain completely private.

Names are provided only to create shared human context.
Names must never be used to assign traits, motives, responsibility, or fault.

YOUR TASK

Produce a shared reflection that both partners can read together.

This reflection must describe how the relationship appears right now,
based only on the emotional substance, clarity, and patterns that are
actually present across both reflections.

SIGNAL AWARENESS (CRITICAL)

Before writing, assess internally:

• Is there sufficient emotional and relational substance?
• Are both partners engaging seriously, or is one clearly disengaged?
• Is the depth roughly balanced, or notably uneven?

If the material is limited, unserious, evasive, or asymmetric,
this must be reflected clearly and honestly.

Do NOT invent depth.
Do NOT stretch meaning.
Do NOT compensate for missing signal with abstraction.

LANGUAGE & TONE

• Write in calm, natural human language
• Avoid therapeutic clichés and inflated psychology
• Avoid meta commentary, analysis labels, or structural markers
• Do not sound like a report or assessment

STRICT RULES

• Do NOT quote or paraphrase specific answers
• Do NOT reveal or imply who said what
• Do NOT assign blame or responsibility
• Do NOT offer advice, reassurance, or next steps
• Do NOT predict the future
• Do NOT repeat the opening or closing ideas inside the reflection body

STRUCTURE (UNLABELED)

Your output must contain exactly three parts, without headings:

1.
Opening context (2–3 sentences)
Briefly frame the overall quality, seriousness, and balance of the inputs.
This framing must appear once only.

1.
Shared reflection
If signal is strong, allow depth and nuance.
If signal is weak or uneven, reflect that limitation directly and plainly.
Do not exceed what the material can support.

1.
Closing boundary (1–2 sentences)
Clarify the limits of the reflection without softening or moralizing.

CORE PRINCIPLES

Honesty is more important than comfort.
Clarity is more important than elegance.

When there is real depth, honor it fully.
When there is not, do not pretend otherwise.

IMPORTANT TECHNICAL CONSTRAINT

Do NOT include headings, labels, section titles, step markers,
markdown symbols, numbering, or formatting cues of any kind
(e.g. ###, STEP, 1), 2), etc.).
If such markers appear, rewrite internally and output clean prose only.
"""

# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return {"message": "ReflectUS backend running"}

@app.post("/create_session")
def create_session(data: CreateSessionInput):
    session_id = str(uuid.uuid4())[:6]
    sessions[session_id] = {
        "A": {
            "name": data.creator_name,
            "answers": [],
            "ready": False
        },
        "B": {
            "name": None,
            "answers": [],
            "ready": False
        },
        "analysis": None,
        "language": data.language or "en"
    }
    return {"session_id": session_id}

@app.post("/join_session")
def join_session(data: JoinSessionInput):
    if data.session_id not in sessions:
        return {"error": "Session not found"}

    sessions[data.session_id]["B"]["name"] = data.name
    return {"role": "B"}

@app.post("/save_answer")
def save_answer(data: SaveAnswerInput):
    if data.session_id not in sessions:
        return {"error": "Session not found"}

    sessions[data.session_id][data.role]["answers"].append(data.answer)
    return {"status": "ok"}

@app.post("/ready")
def ready(data: ReadyInput):
    if data.session_id not in sessions:
        return {"error": "Session not found"}

    session = sessions[data.session_id]
    session[data.role]["ready"] = True

    if (
        session["A"]["ready"]
        and session["B"]["ready"]
        and session["analysis"] is None
    ):
        time.sleep(60)

        session["analysis"] = generate_reflection(
            session["A"]["answers"],
            session["B"]["answers"],
            session["A"]["name"],
            session["B"]["name"],
            session["language"]
        )

    return {"status": "ok"}

# =========================
# WAITING ROOM
# =========================

@app.get("/check_ready/{session_id}")
def check_ready(session_id: str):
    if session_id not in sessions:
        return {"error": "Session not found"}

    s = sessions[session_id]
    return {
        "A_ready": s["A"]["ready"],
        "B_ready": s["B"]["ready"],
        "both_ready": s["A"]["ready"] and s["B"]["ready"]
    }

@app.get("/get_results")
def get_results(session_id: str):
    if session_id not in sessions:
        return {"error": "Session not found"}

    return {"analysis": sessions[session_id]["analysis"]}

# =========================
# OPENAI CALL
# =========================

def generate_reflection(answers_a, answers_b, name_a, name_b, language):
    combined_input = (
        f"{name_a} reflections:\n" + "\n".join(answers_a) + "\n\n"
        f"{name_b} reflections:\n" + "\n".join(answers_b)
    )

    language_map = {
        "en": "English",
        "sl": "Slovenian",
        "hr": "Croatian",
        "sr": "Serbian",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese"
    }

    language_name = language_map.get(language, "English")

    language_instruction = (
        f"The following reflection must be written entirely in {language_name}. "
        f"Do not use any other language."
    )

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            { "role": "system", "content": language_instruction },
            { "role": "system", "content": MEDIATOR_PROMPT },
            { "role": "user", "content": combined_input }
        ],
        max_output_tokens=900
    )

    return response.output_text
