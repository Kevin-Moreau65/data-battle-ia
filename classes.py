from typing import Optional

from pydantic import BaseModel, Field

class Answer(BaseModel):
    content: str = Field()
    isCorrect: bool = Field()
    justification: str = Field()

class Question(BaseModel):
    statement: str = Field(description="The statement of the question in 2 or 3 sentences, it needs to an open question")
    answers: list[Answer] = Field(description="An array of 4 answers, one need to be correct, the others 3 need to be plausible but wrong", min_length=4, max_length=4)

class Generated_Questions(BaseModel):
    questions: list[Question] = Field(description="An array of 10 questions", min_length=10, max_length=10)