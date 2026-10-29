from fastapi import FastAPI, UploadFile, File
import shutil
import os
from rag_pipeline import process_new_pdf, UPLOAD_DIR
from pydantic import BaseModel
from typing import List, Optional
from rag_pipeline import rag_answer

app = FastAPI(title="RAG Q&A API")

class HistoryItem(BaseModel):
    question: str
    answer: str

class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[HistoryItem]] = None

@app.get("/")
def root():
    return {"status": "RAG API is running"}

@app.post("/ask")
def ask(request: QuestionRequest):
    history = [h.dict() for h in request.history] if request.history else None
    result = rag_answer(request.question, history=history)
    return result

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...), username: str = "unknown"):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    num_chunks = process_new_pdf(save_path, file.filename, uploaded_by=username)

    return {
        "filename": file.filename,
        "chunks_added": num_chunks,
        "status": "Document added to knowledge base"
    }