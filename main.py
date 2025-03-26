from ollama import chat
from ollama import ChatResponse
import ollama
model = "mistral-nemo"
embedding_model = "nomic-embed-text"
ollama.pull(model)
ollama.pull(embedding_model)
# response: ChatResponse = chat(model=model, messages=[
#   {
#     'role': 'user',
#     'content': """You are a teacher, you need to create a multiple choice question about basic math (addition and substraction), you will generate a json array of 2 questions with this properties :
#     'question': the name of the question,
#     'answers': an array of 4 elements with a field 'name' which is the answer content, a field isCorrect a boolean that will indicate if the answer is correct or not, and a field 'explanation' that will explain why the answer is correct or not with math logic, one answer will be correct and the other 3 must be plausible but wrong,
#     YOU NEED TO ANSWER WITH A JSON FORMAT
#     """,
#   },
# ])
# print(response.message.content)