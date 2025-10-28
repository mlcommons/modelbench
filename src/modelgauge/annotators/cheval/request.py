from pydantic import BaseModel


class AnnotationRequest(BaseModel):
    annotator: str
    prompt: str = ""
    response: str
