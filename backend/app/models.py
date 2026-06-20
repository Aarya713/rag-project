from pydantic import BaseModel
from typing import List, Optional

class PDFUploadResponse(BaseModel):
    status: str
    num_chunks: int
    num_files: int

class QuestionRequest(BaseModel):
    question: str

class SourceInfo(BaseModel):
    source: str
    page: Optional[int] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []