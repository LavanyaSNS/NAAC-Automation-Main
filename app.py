import os
import re
import logging
from enum import Enum
from datetime import datetime
from typing import List, Optional, Union
from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
import requests
from functools import lru_cache
from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
# Optional: LangSmith debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "OKRionBot"
load_dotenv()
# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="OKRion Chatbot API",
    version="2.0.0",
    description="Improved OKRion Chatbot with enhanced accuracy and support",
    docs_url="/docs"
)
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("DB_NAME", "chatbot")]
# :white_check_mark: Store chat history function
def store_chat(studentID: str, question: str, answer: str) -> None:
    db.chat_history.insert_one({
        "studentId": studentID,
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow()
    })
class IssueType(str, Enum):
    BUG = "Bug"
    FEATURE = "Feature Request"
    SUPPORT = "Technical Support"
    OTHER = "Other"
class ChatRequest(BaseModel):
    student_id: int
    query: str
    name: str
    branch: int
    email: EmailStr
    action: Optional[str] = None
class ChatResponse(BaseModel):
    response: str
    needs_ticket_confirmation: bool
    is_greeting: bool
    chat_history: List[dict] = []
class TicketResponse(BaseModel):
    success: bool
    ticket_id: str
    message: str
    chat_history: List[dict] = []
def is_issue_query(query: str) -> bool:
    issue_keywords = ["error", "bug", "issue", "not working", "problem", "fail", "crash"]
    return any(keyword in query.lower() for keyword in issue_keywords)

class GroqLLM(LLM, BaseModel, Runnable):
    model_name: str = "llama3-70b-8192"  
    temperature: float = 0.4
    api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    base_url: str = "https://api.groq.com/openai/v1/chat/completions"

    def _llm_type(self) -> str:
        return "groq_custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                raise ValueError(f"Groq API Error: {response.status_code} {response.text}")
        except Exception as e:
            raise ValueError(f"Failed to call Groq API: {e}")

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging

logger = logging.getLogger(__name__)

def load_or_create_vectorstore():
    persist_dir = "./chroma_db"
    collection_name = "my_collection"
    embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

    try:
        vectordb = Chroma(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )
        # Check if DB is empty
        if vectordb._collection.count() == 0:
            raise ValueError("Vectorstore is empty")
        print(f"✅ Loaded vectorstore with {vectordb._collection.count()} vectors")
        return vectordb
    except Exception as e:
        logger.warning(f"⚠️ Failed to load vectorstore, recreating: {e}")
        loader = TextLoader("data/all_docx.txt", encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=persist_dir,
            collection_name=collection_name
        )
        vectordb.persist()
        print(f"✅ Created new vectorstore with {len(chunks)} chunks")
        return vectordb

vectorstore = load_or_create_vectorstore()
memory_cache = {}
@lru_cache(maxsize=100)
def get_memory_for_user(user_id: int):
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


llm=GroqLLM()
def get_chain(student_id: int):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=get_memory_for_user(student_id),
        return_source_documents=False,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )
def get_chat_history(student_id: int) -> list:
    history_cursor = db.chat_history.find({"studentId": student_id}).sort("timestamp", -1)
    history = []
    for doc in history_cursor:
        doc["_id"] = str(doc["_id"])
        history.append(doc)
    return history
def create_ticket(name: str, student_id: int, branch: int, email: str, query: str, issue_type: IssueType = IssueType.BUG) -> dict:
    ticket = {
        "name": name,
        "studentId": student_id,
        "branch": branch,
        "email": email,
        "ticketTitle": query[:50],
        "description": query,
        "issueType": issue_type.value,
        "status": "Open",
        "files": [],
        "isActive": True,
        "isDeleted": False,
        "isCompleted": False,
        "timestamp": datetime.utcnow()
    }
    result = db.tickets.insert_one(ticket)
    # :white_check_mark: Store ticket creation chat
    store_chat(student_id, f"Ticket created for: {query[:50]}", f"Ticket ID: {str(result.inserted_id)}")
    return {
        "ticket_id": str(result.inserted_id),
        "success": True
    }
def log_unanswered_query(query: str, student_id: int):
    db.unanswered.insert_one({
        "studentId": student_id,
        "query": query,
        "timestamp": datetime.utcnow()
    })
def handle_general_greetings(query: str) -> Optional[str]:
    normalized = query.strip().lower()
    greetings = {
        "hi": "Hello! How can I assist you today?",
        "hii": "Hello! How can I assist you today?",
        "hello": "Hi there! How can I help you?",
        "bye": "Goodbye! Have a great day!",
        "thanks": "You're welcome! Feel free to ask more questions.",
        "thank you": "You're welcome! Let me know if you need anything else.",
        "how are you": "I'm doing great, thank you! How can I help you today?",
        "good": "Thanks, you can feel free to ask more questions."
    }
    return greetings.get(normalized)  # Match only if the entire query is a greeting
custom_prompt = PromptTemplate.from_template("""
You are an intelligent assistant for a product called OKRion.
Answer ONLY using the CONTEXT provided. If the answer isn't in context, respond:"I couldn't find relevant information in the product documentation."
**Rules:**
- DO NOT make up answers.
- Stick to the provided CONTEXT only.
- If user greets (hi, hello, thanks), respond accordingly.
CONTEXT:
{context}
CHAT HISTORY:
{chat_history}
USER QUESTION:
{question}
ANSWER:
""")
@app.post("/chat/", response_model=Union[ChatResponse, TicketResponse])
async def handle_chat(request: ChatRequest):
    try:
        if request.action == "create_ticket":
            ticket_result = create_ticket(
                name=request.name,
                student_id=request.student_id,
                branch=request.branch,
                email=request.email,
                query=request.query
            )
            return {
                "success": True,
                "ticket_id": ticket_result["ticket_id"],
                "message": "Ticket created successfully",
                "chat_history": get_chat_history(request.student_id)
            }
        greeting_response = handle_general_greetings(request.query)
        if greeting_response:
            # :white_check_mark: Store greeting
            store_chat(request.student_id, request.query, greeting_response)
            return {
                "response": greeting_response,
                "needs_ticket_confirmation": False,
                "is_greeting": True,
                "chat_history": get_chat_history(request.student_id)
            }
        chain = get_chain(request.student_id)
        result = chain({"question": request.query})
        answer = result["answer"]
        needs_ticket_confirmation = (
            is_issue_query(request.query) and
            ("I couldn't find" in answer or "i don't know" in answer.lower())
        )
        if needs_ticket_confirmation:
            log_unanswered_query(request.query, request.student_id)
        # :white_check_mark: Store normal response
        store_chat(request.studen  t_id, request.query, answer)
        return {
            "response": answer,
            "needs_ticket_confirmation": needs_ticket_confirmation,
            "is_greeting": False,
            "chat_history": get_chat_history(request.student_id)
        }
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)