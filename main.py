from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from QA_Bot import BotConfig, QABot

conf = BotConfig()
bot = QABot(conf)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str

@app.post("/chatbot/")
async def get_response(query: Query):
    print(query.message)
    response = process_message(query.message)
    return {"response": response}

def process_message(message: str) -> str:
    answer = bot.forward(message)
    return answer