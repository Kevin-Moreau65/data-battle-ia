from typing import Union

from fastapi import FastAPI
from ollama import chat
from ollama import ChatResponse
from typing_extensions import List
from fastapi import FastAPI, File, UploadFile
import json
import ollama
from pydantic import BaseModel
import os
from helper import extract_json, override_metadata, load_all_pdfs, split_text, vector_store, graphGenerateCorrection, graphGenerateQuestion
model = "mistral-nemo"
ollama.pull(model)
app = FastAPI()
import onnxruntime

print(onnxruntime.get_available_providers())

@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.post("/{subject}/{type}/add_files")
async def add_file(subject: str, type: str,files: List[UploadFile]):
    for file in files:
        try:
            print(file)
            override_metadata(document=file.file, type_of_document=type, subject=subject, output_path=os.path.join(os.getcwd(), "./temp/temp.pdf"))
            documents = load_all_pdfs(directory=os.path.join(os.getcwd(), "./temp"), subject=subject, type_of_document=type)
            splitted_doc = split_text(documents)
            await vector_store.aadd_documents(documents=splitted_doc)
            os.remove(os.path.join(os.getcwd(), "./temp/temp.pdf"))
        except Exception as e: print(e)
    return {"name": files[0].filename}
@app.get("/{subject}/generate_question")
def read_promt(subject: str):
    response = graphGenerateQuestion.invoke({
    "question": 
    f"""
    You are teacher in charge to create a multiple choice question about {subject} using the context of this message, you need to generate 10 questions, every questions need to have context with a company having an issue or a question about the subject. Every answers need to have a content of 2 - 3 sentences and a justification of why is it correct or not.
""", 
    "subject": subject
    })
    try:
        return {"result": json.loads(extract_json(response["answer"].replace("\n", "").replace("\\", "")))}
    except Exception as error:
        print(error)
        return {"result": response["answer"]}
class Correction(BaseModel):
    question: str
    answer: str
@app.post("/{subject}/correct")
def read_promt(subject: str, item: Correction):
    response = graphGenerateCorrection.invoke({
    "question": 
    f"""
    You are teacher in charge to correct a question about {subject} using the context of this message, for the question : "{item.question}" 
    a student answered : "{item.answer}"
    If the answer of the student is correct is not justified, then it's incorrect
    Is it correct and why ?
""", 
    "subject": subject
    })
    try:
        return {"result": json.loads(extract_json(response["answer"].replace("\n", "").replace("\\", "")))}
    except Exception as error:
        print(error)
        return {"result": response["answer"]}