from openai import AsyncOpenAI
from agents import Agent,  Runner,run_demo_loop, function_tool, RunContextWrapper, set_default_openai_client, set_default_openai_api, set_tracing_disabled
import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

set_tracing_disabled(True)

gemini_api_key = os.getenv('GEMINI_API_KEY')

client = AsyncOpenAI(api_key=gemini_api_key,
                     base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


set_default_openai_api("chat_completions")
set_default_openai_client(client)

@dataclass
class UserContext:
    Name: str

def dynamic_instructions(
        context: RunContextWrapper[UserContext], agent: Agent[UserContext]
) -> str:
    return f"The users name is {context.context.Name}. Help me with their questions."

agent = Agent[UserContext](
    name = "Helpful Agent",
    instructions=dynamic_instructions,
    model="gemini-2.0-flash"
)

query = input("Enter query: ")

result = Runner.run_sync(
    agent,query, context=UserContext(Name="Ali")
)

print(result.final_output)