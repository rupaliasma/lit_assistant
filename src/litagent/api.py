"""HTTP API: ``POST /ask`` with a question, get back a cited answer.

Set ``LITAGENT_API_KEY`` to require clients to send it in an ``X-API-Key`` header.
"""

import logging
import os
import secrets

import anthropic
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from litagent import __version__
from litagent.agents import LiteratureAssistant
from litagent.llm import RefusalError

log = logging.getLogger(__name__)


class Question(BaseModel):
    question: str = Field(min_length=10, max_length=1000, examples=["How does GRAPPA differ from SENSE?"])


class Source(BaseModel):
    pmid: str
    title: str
    journal: str
    year: str
    authors: list[str]
    doi: str | None
    url: str


class Check(BaseModel):
    claim: str
    pmid: str
    status: str
    reason: str


class AnswerOut(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    checks: list[Check]
    citation_precision: float | None
    uncited: list[str]
    revised: bool
    papers_read: list[str]
    usage: dict


def create_app(assistant: LiteratureAssistant | None = None) -> FastAPI:
    app = FastAPI(title="Literature assistant for MRI reconstruction", version=__version__)
    app.state.assistant = assistant
    api_key = os.environ.get("LITAGENT_API_KEY")

    def check_key(x_api_key: str | None = Header(default=None)):
        if api_key and not (x_api_key and secrets.compare_digest(x_api_key, api_key)):
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    def get_assistant() -> LiteratureAssistant:
        if app.state.assistant is None:  # created lazily so the app can start without credentials
            app.state.assistant = LiteratureAssistant()
        return app.state.assistant

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    # A plain `def` endpoint runs in FastAPI's thread pool, so one slow question does not block others.
    @app.post("/ask", response_model=AnswerOut, dependencies=[Depends(check_key)])
    def ask(body: Question, assistant: LiteratureAssistant = Depends(get_assistant)):
        try:
            return assistant.ask(body.question.strip()).to_dict()
        except RefusalError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        except anthropic.RateLimitError as err:
            raise HTTPException(status_code=429, detail="language model rate limit reached, retry later") from err
        except anthropic.APIStatusError as err:
            log.exception("language model error")
            raise HTTPException(status_code=502, detail=f"language model error ({err.status_code})") from err
        except anthropic.APIConnectionError as err:
            raise HTTPException(status_code=503, detail="language model unreachable") from err
        except httpx.HTTPError as err:
            log.exception("upstream request failed")
            raise HTTPException(
                status_code=502, detail="upstream request failed (PubMed, Europe PMC or the model server)"
            ) from err

    return app


app = create_app()
