import asyncio
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client, set_tracing_disabled, run_demo_loop
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

set_tracing_disabled(True)

client = AsyncOpenAI(api_key=gemini_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

set_default_openai_api("chat_completions")
set_default_openai_client(client)

query = input("Enter Query: ")

async def main():
    agent = Agent(name="Joker",
                  instructions="You are a helpful assistant",
                  model="gemini-2.0-flash"
                  )
    # await run_demo_loop(agent)

    result = Runner.run_streamed(agent, query)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
        


asyncio.run(main())