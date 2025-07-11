from agents import Agent, Runner,  set_tracing_disabled, set_default_openai_client, set_default_openai_api
from openai import AsyncOpenAI
import os
from agents.model_settings import ModelSettings
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

set_tracing_disabled(True)

gemini_api_key = os.getenv('GEMINI_API_KEY')

    
client = AsyncOpenAI(api_key=gemini_api_key,
                         base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

   
set_default_openai_api("chat_completions")
set_default_openai_client(client)

    # Initialize the agent with its name and instructions
agent = Agent(name="Assistant", instructions="You are a helpful assistant", model='gemini-2.0-flash')

@cl.on_message
async def main(message: cl.Message):
    """
    Handle incoming messages. This method is triggered by each user message.
    """
    
    

    # Inform the user that the assistant is thinking
    thinking_msg = cl.Message(content="Thinking...")
    await thinking_msg.send()

    # Process the query through the agent using a sync runner
    query = message.content  # Ensure we're using just the message content for the query
    result = Runner.run_sync(agent, query)
    response = result.final_output

    # Update the message with the final response
    thinking_msg.content = response
    await thinking_msg.update()

    # Send the final response to the user
    await thinking_msg.send()

    
# query = input("Enter query:  ")

# print(result.final_output)