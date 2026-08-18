# service.py — LLMNode: natural language → RootNode tree_query plan
#
# Start (Ollama must already be running with the model pulled):
#   ollama pull llama3.2:3b
#   uvicorn service:app --host 0.0.0.0 --port 8000
#
# RootNode .env:
#   TREE_QUERY_LLM_URL=http://llmnode:8000/parse
#   TREE_QUERY_LLM_API_KEY=   # optional, must match LLMNODE_API_KEY

from __future__ import annotations

import json
import os
from typing import Any

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
API_KEY = os.getenv("LLMNODE_API_KEY", "").strip()
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))

ALLOWED_INTENTS = frozenset(
    {
        "resolve_kinship",
        "count_children",
        "list_children",
        "person_facts",
        "relation_between",
    }
)
ALLOWED_STEPS = frozenset({"father", "mother", "spouse", "child", "sibling"})

STEP_ALIASES = {
    "father": "father",
    "vater": "father",
    "papa": "father",
    "dad": "father",
    "mother": "mother",
    "mutter": "mother",
    "mama": "mother",
    "mom": "mother",
    "mum": "mother",
    "spouse": "spouse",
    "partner": "spouse",
    "ehepartner": "spouse",
    "ehemann": "spouse",
    "ehefrau": "spouse",
    "husband": "spouse",
    "wife": "spouse",
    "child": "child",
    "kind": "child",
    "sohn": "child",
    "tochter": "child",
    "son": "child",
    "daughter": "child",
    "sibling": "sibling",
    "geschwister": "sibling",
    "bruder": "sibling",
    "schwester": "sibling",
    "brother": "sibling",
    "sister": "sibling",
}

SYSTEM_PROMPT = """You convert genealogy questions into a JSON plan. Output JSON only, no markdown.
Schema:
{
  "intent": "resolve_kinship" | "count_children" | "list_children" | "person_facts" | "relation_between",
  "kinship_path": ["father" | "mother" | "spouse" | "child" | "sibling"],
  "person_name": "",
  "target_name": ""
}
Rules:
- Questions about "my/ich/mein/meine" use empty person_name (the tree starting person).
- Never invent numeric ids.
- Maternal grandmother = ["mother","mother"]. Paternal grandfather = ["father","father"].
- kinship_path walks from the subject toward the relative.
- relation_between: put the two people into person_name and target_name, empty kinship_path.
- count_children / list_children: set kinship_path to the person whose children are asked.
- person_facts: birth/death/marriage of the resolved person.
"""

app = FastAPI(title="LLMNode-Parse-Service")


class ParseRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


def _require_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if (x_api_key or "").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Ungültiger API-Schlüssel.")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        inner = [line for line in lines if not line.strip().startswith("```")]
        raw = "\n".join(inner).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None


def _normalize_path(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        tokens = [p.strip().lower() for p in raw.replace(";", ",").split(",") if p.strip()]
    elif isinstance(raw, (list, tuple)):
        tokens = [str(item).strip().lower() for item in raw if str(item).strip()]
    else:
        raise ValueError("kinship_path muss eine Liste oder ein Text sein.")

    out: list[str] = []
    for step in tokens:
        canonical = STEP_ALIASES.get(step)
        if not canonical or canonical not in ALLOWED_STEPS:
            raise ValueError(f"Unbekannter Verwandtschaftsschritt: {step}")
        out.append(canonical)
    if len(out) > 8:
        raise ValueError("Verwandtschaftspfad ist zu lang.")
    return out


def _sanitize_plan(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop model-invented ids; keep a RootNode-compatible plan."""
    intent = str(raw.get("intent") or "").strip()
    if intent not in ALLOWED_INTENTS:
        raise ValueError(
            "Unbekannte Anfrageart. Erlaubt: " + ", ".join(sorted(ALLOWED_INTENTS))
        )
    return {
        "intent": intent,
        "kinship_path": _normalize_path(raw.get("kinship_path")),
        "person_name": str(raw.get("person_name") or "").strip(),
        "target_name": str(raw.get("target_name") or "").strip(),
        "anchor": "starting_individual",
        "person_id": None,
        "target_id": None,
    }


def _plan_from_ollama_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Ungültige Ollama-Antwort.")

    content = ""
    message = payload.get("message")
    if isinstance(message, dict):
        raw_content = message.get("content")
        if isinstance(raw_content, dict):
            return _sanitize_plan(raw_content)
        content = str(raw_content or "")
    elif payload.get("response"):
        content = str(payload.get("response") or "")
    elif "intent" in payload:
        return _sanitize_plan(payload)

    plan = _extract_json_object(content)
    if not plan:
        raise ValueError("Das Sprachmodell lieferte keinen gültigen JSON-Plan.")
    return _sanitize_plan(plan)


def _call_ollama(question: str) -> dict[str, Any]:
    url = f"{OLLAMA_URL}/api/chat"
    body = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    try:
        response = requests.post(url, json=body, timeout=OLLAMA_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise HTTPException(
            status_code=504, detail="Zeitüberschreitung beim Sprachmodell."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama nicht erreichbar unter {OLLAMA_URL}. Läuft der Dienst?",
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=502, detail=f"Ollama-Anfrage fehlgeschlagen: {exc}"
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama-HTTP-Fehler ({response.status_code}).",
        )

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502, detail="Ungültige JSON-Antwort von Ollama."
        ) from exc

    try:
        return _plan_from_ollama_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "model": OLLAMA_MODEL, "ollama": OLLAMA_URL}


@app.post("/parse")
def parse_question(
    payload: ParseRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """
    Nimmt eine Alltagsfrage und gibt einen strukturierten tree_query-Plan zurück.
    RootNode führt den Plan aus — dieses Service rechnet keine Verwandtschaft.
    """
    _require_api_key(x_api_key)
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Keine Frage angegeben.")

    plan = _call_ollama(question)
    return JSONResponse(content={"success": True, "plan": plan})
