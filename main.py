from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from llm import answer_question

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://jasonherngwang.com",
    "https://www.jasonherngwang.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    id: str
    sender: str
    message: str


class QueryWithHistory(BaseModel):
    query: str
    history: list[Message]


@app.get("/")
async def root():
    return {"message": "Hello world!"}


@app.get("/health")
def health():
    return "OK"


@app.post("/query")
@limiter.limit("6/minute")
async def send_query(query_with_history: QueryWithHistory, request: Request):
    try:
        response = answer_question(query_with_history.query, query_with_history.history)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
