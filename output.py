import asyncio
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
from pydantic import BaseModel

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

set_tracing_disabled(True)

client = AsyncOpenAI(api_key=gemini_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

set_default_openai_api("chat_completions")
set_default_openai_client(client)

query = input("Enter Query: ")

class Event_info(BaseModel):
    name: str
    date: str
    time: str


agent = Agent(name="calender Event", instructions="You are a Event information extractor",
              model="gemini-2.0-flash",
               output_type=Event_info )

result = Runner.run_sync(agent, query)

print(result.final_output)