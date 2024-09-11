import itertools
import multiprocessing
import multiprocessing.pool
import os
from typing import Sequence

from fastapi import FastAPI, Depends, HTTPException  # type: ignore
from fastapi.security import APIKeyHeader  # type: ignore
from pydantic import BaseModel

from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.config import load_secrets_from_config
from modelgauge.load_plugins import load_plugins
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import PromptResponseSUT
from modelgauge.sut_registry import SUTS
from modelgauge.suts.together_client import CHAT_MODELS

"""
  Simple API server for modelgauge functionality. Currently used just for interviews.
  
  Start it up with something like `fastapi run modelgauge/api_server.py`
  
  To use it, GET / will show the list of available SUTs. Then you can POST / with
  something like:
  
  ```
  {
    "prompts": [{"text": "What's your name?","options": {"max_tokens": 50}}],
    "suts":["llama-2-7b-chat"]
  }
  ```
Multiple SUTs are allowed, and are run in parallel.
"""

load_plugins()

secrets = load_secrets_from_config()

suts: dict[str, PromptResponseSUT] = {
    sut_uid: SUTS.make_instance(sut_uid, secrets=secrets)  # type:ignore
    for sut_uid in CHAT_MODELS.keys()
}

annotators: dict[str, CompletionAnnotator] = {
    sut_uid: ANNOTATORS.make_instance(sut_uid, secrets=secrets)  # type:ignore
    for sut_uid in [i[0] for i in ANNOTATORS.items()]
}

print(f"got suts {suts} and annotators {annotators}")


class ProcessingRequest(BaseModel):
    prompts: Sequence[TextPrompt]
    suts: Sequence[str]
    annotators: Sequence[str] = []


SECRET_KEY = os.getenv("SECRET_KEY")
assert SECRET_KEY, "must set SECRET_KEY environment variable"
app = FastAPI()


@app.get("/")
async def get_options():
    return {"suts": list(suts.keys()), "annotators": list(annotators.keys())}


def process_sut_item(prompt: TextPrompt, sut_key: str):
    sut = suts[sut_key]
    s_req = sut.translate_text_prompt(prompt)
    s_resp = sut.translate_response(s_req, sut.evaluate(s_req))
    return {"sut": sut.uid, "prompt": prompt, "sut_response": s_resp}


def process_annotation(result: dict, annotator_keys: Sequence[str]):
    result["annotations"] = {}
    for key in annotator_keys:
        annotator = annotators[key]
        a_req = annotator.translate_request(
            PromptWithContext(prompt=result["prompt"], source_id="whatever, man"),
            result["sut_response"].completions[0],
        )
        result["annotations"][key] = annotator.translate_response(
            a_req, annotator.annotate(a_req)
        )
    return result


auth_header = APIKeyHeader(name="x-key")


async def process_work_items(function, work_items):
    if not work_items:
        return []
    pool = multiprocessing.pool.ThreadPool(len(work_items))
    return pool.starmap(function, work_items)


@app.post("/")
async def process_sut_request(req: ProcessingRequest, key: str = Depends(auth_header)):
    if key != SECRET_KEY:
        raise HTTPException(401, "not authorized; send x-key header")
    for sut in req.suts:
        if not sut in suts:
            raise HTTPException(422, f"sut {sut} not found")

    sut_work_items = list(itertools.product(req.prompts, req.suts))  # type:ignore
    sut_results = await process_work_items(process_sut_item, sut_work_items)

    if req.annotators:
        annotator_work_items = [
            [sut_result, req.annotators] for sut_result in sut_results
        ]
        await process_work_items(process_annotation, annotator_work_items)

    return {"response": sut_results}
