"""Matching helpers for routing questions to the right knowledge entry."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Tuple

from dotenv import load_dotenv
import google.genai as genai
from openai import OpenAI

from .db import fetch_stock_rows, get_postgres_connection

__all__ = [
    "detect_escalation",
    "engine_match",
    "summarize_conversation",
    "find_relevant_history_reply",
    "build_product_enquiry_prompt",
    "fetch_stock_rows",
    "get_postgres_connection",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
ACCESS_ALLOWED = os.getenv("ENGINE_MATCHING_ENABLED", "true").lower() in {"1", "true", "yes"}
DEFAULT_STOCK_TABLE = os.getenv("STOCK_TABLE", "shopify_variant_new")


def _ensure_access_allowed() -> None:
    if not ACCESS_ALLOWED:
        raise PermissionError(
            "Access to engine_matching is disabled. Set ENGINE_MATCHING_ENABLED=true to enable."
        )


def _get_gemini_client() -> genai.Client:
    _ensure_access_allowed()
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Set the GEMINI_API_KEY environment variable before using the Gemini model."
        )
    
    return genai.Client(api_key=GEMINI_API_KEY)


def _get_openai_client() -> OpenAI:
    _ensure_access_allowed()
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Set the OPENAI_API_KEY environment variable before using the OpenAI model."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def _build_default_stock_schema() -> str:
    """Return the default SQL schema definition for the stock table."""

    return (
        "CREATE TABLE shopify_variant_new (\n"
        "  product_id        BIGINT NOT NULL,\n"
        "  variant_id        BIGINT NOT NULL,\n"
        "  color             TEXT,\n"
        "  spec              TEXT,\n"
        "  condition         TEXT,\n"
        "  price             NUMERIC(12,2),\n"
        "  handle            TEXT,\n"
        "  vendor            TEXT,\n"
        "  product_type      TEXT,\n"
        "  tenure            TEXT\n"
        ");"
    )


def _build_prompt(
    user_question: str,
    options: Iterable[str],
    conversation_summary: str = "",
) -> str:
    print("Helloconversation_summary ", conversation_summary)
    summary_section = ""
    if conversation_summary:
        summary_section = (
            "\nPrevious conversation summary:\n"
            f"\"\"\"\n{conversation_summary}\n\"\"\"\n\n"
            "Use this summary to maintain context. If it is unrelated, ignore it and continue normally.\n"
        )
        print("HelloSummarySection: ", summary_section)

    return f"""
You are a Compasia customer service representative for a leading second-hand HP device seller.

Your task is to:
1) Understand the customer's request
2) Decide whether it should be matched to an existing knowledge-base question
3) Or handled as a live support case

Conversation summary:
{summary_section}

List of possible questions:
{list(options)}

User question:
"{user_question}"

IMPORTANT RULES (follow strictly):

1. If the conversation summary OR user question shows the user wants to buy a phone, asks about phone functionality, or otherwise makes a product enquiry, return exactly:
   "PRODUCT_ENQUIRE"

2. If the conversation summary OR user question contains personal details such as:
   - Full name
   - IC number
   - Order ID
   - Contract ID
   - Phone number
   - Email address

   Then return exactly:
   "TICKET_LOGGED"

3. If NO personal details are found and you find a strong match with the knowledge base,
   return ONLY the exact matching question from the List of possible questions.

4. If no suitable knowledge-base match exists and it should remain a live conversation,
   return exactly:
   "NO_MATCH"

5. Do NOT explain your reasoning.
6. Do NOT return anything other than:
   - "PRODUCT_ENQUIRE"
   - One exact question from the list
   - "NO_MATCH"
   - "TICKET_LOGGED"

"""




def build_product_enquiry_prompt(
    user_message: str,
    stock_table_schema: str,
    conversation_summary: str = "",
) -> str:
    """Return a Gemini-ready prompt used when a product enquiry is detected."""

    summary_section = ""
    if conversation_summary:
        summary_section = (
            "\nConversation summary:\n"
            f"\"\"\"\n{conversation_summary}\n\"\"\"\n\n"
            "Use the summary to avoid repeating prior proposals and focus on unmet needs.\n"
        )

    schema = stock_table_schema.strip() or _build_default_stock_schema()
    system_prompt = f"""
You are a CompAsia sales agent.

Your role is to act like a friendly, human sales consultant.
Help customers choose a suitable device based on their needs.

Rules:
- Recommend ONLY from available stock stored in the database.
- First, understand the table structure from the schema.
- Then produce a SQL query (script) that retrieves matching stock.
- Use PostgreSQL syntax and the schema below.
- Use the exact table name from the schema.
- Select only relevant columns needed to recommend a single best option.
- Always limit the result set to the top 5 rows.
- Prefer lower price and better condition (Excellent > Fair).
- If the user specifies exact model, storage, and color, filter for exact matches.
- Return ONLY the SQL query (no explanations, no markdown).

{summary_section}

Database schema (DDL):
{schema}
"""

    print("Hello_schema:", schema)
    return (
        f"{system_prompt}\n\n"
        f"User message:\n{user_message}\n\n"
        "Generate the SQL query now."
    )


def detect_escalation(user_question: str) -> Tuple[bool, str]:
    _ensure_access_allowed()
    text = user_question.lower()

    print("HelloText: ", text)
    greeting_keywords = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "good day",
    ]
    ticket_keywords = [
        "ticket",
        "support ticket",
        "log a ticket",
        "open a ticket",
        "raise a ticket",
        "submit a ticket",
    ]
    human_keywords = [
        "human",
        "agent",
        "representative",
        "support person",
        "staff",
        "talk to",
        "speak to",
        "live agent",
        "real person",
    ]
    
    if any(text == term or text.startswith(f"{term} ") for term in greeting_keywords):
        return True, "How can I help you today?"

    if any(term in text for term in ticket_keywords):
        return True, "I can help you log a support ticket. Please provide more details about the issue so I can submit it."

    if any(term in text for term in human_keywords):
        return True, "I'll connect you with a human representative. Please share your preferred contact details so we can follow up."
    

    return False, ""


def engine_match(
    user_question: str,
    knowledge_df,
    provider: str = "gemini",
    conversation_summary: str = "",
    stock_table_schema: str = "",
) -> Tuple[str, float, object | None]:
    _ensure_access_allowed()
    keyword_series = knowledge_df["keyword"].astype(str).str.strip()
    options = keyword_series.tolist()
    prompt = _build_prompt(user_question, options, conversation_summary)
    provider_name = provider.lower()

    if provider_name == "gemini":
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents=prompt,
        )
        match = response.text.strip()
        score = 0.0
    elif provider_name == "openai":
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Respond ONLY with a JSON object: {\"match\": <string>, \"score\": <number between 0 and 1>}",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        match = str(parsed.get("match", "NO_MATCH")).strip()
        score = float(parsed.get("score", 0))
    else:
        raise ValueError("provider must be either 'gemini' or 'openai'")

    if match == "PRODUCT_ENQUIRE":
        schema = stock_table_schema or _build_default_stock_schema()
        sales_prompt = build_product_enquiry_prompt(
            user_question,
            schema,
            conversation_summary=conversation_summary,
        )
        client = _get_gemini_client()
        sales_response = client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents=sales_prompt,
        )
        sql_query = sales_response.text.strip()
        matched_row = fetch_stock_rows(sql_query)
        return match, score, matched_row

    normalized_match = match.lower()
    matched_rows = knowledge_df[
        keyword_series.str.lower() == normalized_match
    ]
    matched_row = matched_rows.iloc[0] if not matched_rows.empty else None


    return match, score, matched_row


def find_relevant_history_reply(
    conversation_history: Iterable[str],
    current_question: str,
    provider: str = "gemini",
) -> str | None:
    """Use the configured LLM to locate a relevant prior assistant reply.

    The function inspects the provided ``conversation_history`` (oldest to
    newest), removes empty entries, and drops the trailing message when it
    duplicates ``current_question``. It then asks the LLM to return the most
    recent assistant response that directly answers ``current_question``. When
    no suitable reply exists, ``None`` is returned.
    """

    _ensure_access_allowed()

    cleaned_history = [
        str(message).strip() for message in conversation_history if str(message).strip()
    ]
    trimmed_question = current_question.strip()
    if cleaned_history and trimmed_question:
        last_entry = cleaned_history[-1].lower()
        if last_entry == trimmed_question.lower():
            cleaned_history = cleaned_history[:-1]

    if not cleaned_history or not trimmed_question:
        return None

    transcript = "\n".join(cleaned_history)
    prompt = (
        "You are reviewing a previous support conversation (oldest to newest).\n"
        "User just asked the latest question below. If a prior assistant reply in the"
        " transcript already answers it, return that assistant reply verbatim."
        " If nothing in history answers it, return NO_MATCH. Do not invent new"
        " information.\n\n"
        f"Latest user question: \"{trimmed_question}\"\n\n"
        f"Conversation transcript:\n\"\"\"\n{transcript}\n\"\"\""
    )

    provider_name = provider.lower()

    if provider_name == "gemini":
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents=prompt,
        )
        match = response.text.strip()
    elif provider_name == "openai":
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return ONLY a JSON object like {\"reply\": <string>} where"
                        " reply is either the most relevant past assistant message or"
                        " \"NO_MATCH\"."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        match = str(parsed.get("reply", "NO_MATCH")).strip()
    else:
        raise ValueError("provider must be either 'gemini' or 'openai'")

    if not match or match.lower() == "no_match":
        return None

    return match


def summarize_conversation(
    conversation_history: Iterable[str],
    provider: str = "gemini",
    previous_summary: str = "",
) -> str:
    """Summarize a conversation transcript using the specified provider."""

    _ensure_access_allowed()

    history_lines = [str(message).strip() for message in conversation_history if str(message).strip()]
    cleaned_previous_summary = previous_summary.strip()
    if not history_lines:
        return cleaned_previous_summary

    transcript = "\n".join(history_lines)
    summary_section = ""
    # if cleaned_previous_summary:
    summary_section = f"Previous summary:\n\"\"\"\n{previous_summary}\n\"\"\"\n\n"
    prompt = (
        "You are assisting a Compasia support agent. Update the existing summary with the new conversation "
        "messages below. Provide a concise, neutral summary in two to three sentences, focusing on the "
        "customer's request and any guidance already provided. Avoid repeating sentences. "
        "If the user shows interest in buying a phone or asks about phone functionality, append a final line "
        "at the bottom of the summary to capture desired phone type and specs, using the format "
        "\"Desired phone type/specs: <details or needed>\".\n\n"
        f"{summary_section}"
        f"New conversation transcript:\n\"\"\"\n{transcript}\n\"\"\""
    )

    print("HelloPrompt: ", prompt)

    provider_name = provider.lower()

    if provider_name == "gemini":
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()

    if provider_name == "openai":
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Update the provided conversation summary in plain text (no JSON) using no more than "
                        "three sentences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    raise ValueError("provider must be either 'gemini' or 'openai'")


if __name__ == "__main__":
    raise SystemExit(
        "engine_matching should be imported from trusted code instead of executed directly."
    )
