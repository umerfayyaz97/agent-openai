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
class User:
  user_id: int



@function_tool  
async def get_user_info(ctx: RunContextWrapper[User]) -> str:
    
    """Fetches the user personal information to personalize responses. Whenver you require user personal info. call this function

    Args:
        id: The user unique indentifier
    """
    id = ctx.context.user_id
    if id == 1:
        user_info = "User name is Ali. He is 19 years old. He is a Agentic AI Engineer by profession. He likes playing Cricket. Not a premium user"
    elif id == 2:
        user_info = "User name is Usman. He is 30 years old. He is a doctor by profession. He likes mountains. Premium user"
    else:
        user_info = "user not found"

    return user_info

agent = Agent[User](
    name="Assistant",
    instructions="You are an expert of agentic AI.",
    model="gemini-2.0-flash",
    tools=[get_user_info]
)

query = input("Enter the query: ")

result = Runner.run_sync(
    agent,
    query,
    context=User(user_id = 2 )
)

print(result.final_output)