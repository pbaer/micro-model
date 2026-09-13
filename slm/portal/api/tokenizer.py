from __future__ import annotations

import anyio
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/tokenizers")


class EncodeRequest(BaseModel):
    text: str = ""
    mode: str = "raw"  # raw | document | chat
    messages: list[dict] | None = None
    add_generation_prompt: bool = False


@router.get("")
def tags(request: Request) -> list[dict]:
    return request.app.state.tokenizers.tags()


@router.post("/{tag}/encode")
async def encode(request: Request, tag: str, body: EncodeRequest) -> dict:
    reg = request.app.state.tokenizers
    try:
        return await anyio.to_thread.run_sync(lambda: reg.encode(tag, body.text, body.mode, body.messages, body.add_generation_prompt))
    except KeyError:
        raise HTTPException(404, "unknown tokenizer") from None


@router.get("/{tag}/vocab")
async def vocab(request: Request, tag: str, q: str = "", limit: int = 100) -> list[dict]:
    reg = request.app.state.tokenizers
    return await anyio.to_thread.run_sync(lambda: reg.vocab_search(tag, q, min(limit, 500)))


@router.get("/{tag}/token/{i}")
def token(request: Request, tag: str, i: int) -> dict:
    return request.app.state.tokenizers.token(tag, i)
