import json
from pypdf import PdfWriter, PdfReader
import pypdf
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
import re
from classes import Corrected_Question, Generated_Questions
from main import embedding_model
import random
def override_metadata(document, type_of_document, subject, output_path):
    reader = PdfReader(stream=document)
    writer = PdfWriter(reader)
    writer.metadata = None
    writer.add_metadata(
    {
        "/Type": type_of_document,
        "/Subject": subject,
    }
    )
    with open(output_path, "wb") as f:
        writer.write(f)
def load_all_pdfs(directory,type_of_document, subject):
    """Load all PDFs from the specified directory"""
    print("Loading PDFs from directory...")
    all_documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            try:
                file_path = os.path.join(directory, filename)
                print(f"Loading {filename}...")
                loader = PyPDFLoader(file_path, mode="single")
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e: print(e)
    print(f"Total PDFs loaded: {len(all_documents)}")
    return all_documents
llm = ChatOllama(
    model="mistral-nemo",
    temperature=0.7,
    max_retries=2,
)
structured_llm = llm.with_structured_output(Generated_Questions)
structured_llm_correction = llm.with_structured_output(Corrected_Question)
embeddings = OllamaEmbeddings(model=embedding_model)
def split_text(documents):
    """Split the documents into chunks"""
    print("Splitting documents into chunks...")
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=384)
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks
vector_store = Chroma(embedding_function=embeddings,persist_directory="./chroma_langchain_db")
prompt = hub.pull("rlm/rag-prompt")
# Définir l'état de l'application avec un dictionnaire typé
class State(TypedDict):
    question: str  # La question posée par l'utilisateur
    subject: str # Le sujet
    
    context: List[Document]  # Contexte sous forme de liste de documents récupérés
    answer: str # La réponse générée pour la question
def retrieveGenerateQuestion(state: State):
    # Recherche de documents similaires à la question dans la base de données
    retrieved_courses_with_scores = vector_store.similarity_search_with_score(state["question"], k=50, filter={"$and": [{"subject": state["subject"]}, {"type": "courses"}]})
    retrieved_courses_with_scores = random.sample(retrieved_courses_with_scores, 10)
    retrieved_questions_with_scores = vector_store.similarity_search_with_score(state["question"], k=20, filter={"$and": [{"subject": state["subject"]}, {"type": "questions"}]})
    retrieved_questions_with_scores = random.sample(retrieved_questions_with_scores, 5)
    retrieved_docs_with_scores = []
    retrieved_docs_with_scores.extend(retrieved_courses_with_scores)
    retrieved_docs_with_scores.extend(retrieved_questions_with_scores)
    # Extraire les documents récupérés
    retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
    
    # Extraire les scores des documents récupérés
    # distances = [score for doc, score in retrieved_docs_with_scores]
    #print("Distances des documents récupérés :", distances)  # Affichage optionnel des distances pour vérification
    
    return {"context": retrieved_docs}
def retrieveGenerateCorrection(state: State):
    # Recherche de documents similaires à la question dans la base de données
    retrieved_courses_with_scores = vector_store.similarity_search_with_score(state["question"], k=20, filter={"$and": [{"subject": state["subject"]}, {"type": "courses"}]})
    retrieved_docs_with_scores = []
    retrieved_docs_with_scores.extend(retrieved_courses_with_scores)
    # Extraire les documents récupérés
    retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
    
    # Extraire les scores des documents récupérés
    # distances = [score for doc, score in retrieved_docs_with_scores]
    #print("Distances des documents récupérés :", distances)  # Affichage optionnel des distances pour vérification
    
    return {"context": retrieved_docs}
# Fonction pour générer la réponse à la question en fonction du contexte
def generate(state: State):
    # Joindre le contenu des documents du contexte pour former un bloc de texte
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # Générer un message à partir du prompt avec la question et le contexte
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    # print(messages)
    # Invoker le modèle de langage pour générer la réponse
    response = structured_llm_correction.invoke(f"""[INST] Instruction : Answer the question based on the context
                          {docs_content}
                        ### QUESTION : {state["question"]}
                          [/INST]""")
    return {"answer": response}
def generate_correction(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    response = structured_llm.invoke(f"""[INST] Instruction : Answer the question based on the context
                          {docs_content}
                        ### QUESTION : {state["question"]}
                          [/INST]""")
# Compiler l'application et la tester
graph_builder = StateGraph(State).add_sequence([retrieveGenerateQuestion, generate])  # Ajouter les étapes à l'application
graph_builder.add_edge(START, "retrieveGenerateQuestion")  # Définir la première étape du graph comme "retrieve"
graphGenerateQuestion = graph_builder.compile() # Compiler le graph pour l'exécution
graph_builder = StateGraph(State).add_sequence([retrieveGenerateCorrection, generate_correction])  # Ajouter les étapes à l'application
graph_builder.add_edge(START, "retrieveGenerateCorrection")  # Définir la première étape du graph comme "retrieve"
graphGenerateCorrection = graph_builder.compile() # Compiler le graph pour l'exécution
def extract_json(str: str):
    matchedText = re.search(r"{([^;]*)}", str, re.IGNORECASE).group(0)
    return matchedText