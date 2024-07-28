"""Add endpoints to IPA using FastAPI."""

from typing import Any

from aif.ipa import Request, Response, ipa


@ipa.app.get("/hello")
@ipa.unauthenticated
async def hello(request: Request, response: Response) -> Any:
    """Be greeted."""
    return {"hello": "world"}
