from agents import Agent, Runner,  set_tracing_disabled, set_default_openai_client, set_default_openai_api
from openai import AsyncOpenAI
import os
from agents.model_settings import ModelSettings
from dotenv import load_dotenv

load_dotenv()

set_tracing_disabled(True)

gemini_api_key = os.getenv('GEMINI_API_KEY')

client = AsyncOpenAI(api_key=gemini_api_key,
                     base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


set_default_openai_api("chat_completions")
set_default_openai_client(client)

# model_setting = ModelSettings(
#     temperature=0.8,
#     top_p=0.7,
    
# )

agent = Agent(name="Assistant", instructions="You are a helpful assistant", model='gemini-2.0-flash'
              )

query = input("Enter query:  ")

result = Runner.run_sync(agent, query, max_turns=2)



print(result.final_output)