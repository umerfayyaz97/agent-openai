from agents import Agent, Runner,  set_tracing_disabled, set_default_openai_client, set_default_openai_api, function_tool, handoff
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
model = "gemini-2.0-flash"



WebDev : Agent = Agent(
    name= "WebDev Agent",
    instructions="You are a web development agent, you answer web development related questions",
    model=model,
    handoff_description="Web Development Expert"
)

#Agent as tools
MobileDev : Agent = Agent(
    name= "MobileDev Agent",
    instructions="You are a Mobile App Development agent, you answer Mobile App development related questions",
    model=model

)

#Agent as tools
DevOps: Agent = Agent(
    name= "DevOps Agent",
    instructions="You are a DevOps agent, you answer DevOps related questions",
    model=model
)

OpenAIAgent : Agent = Agent(
    name= "OpenAI Agent",
    instructions="You are an OpenAI agent, you answer OpenAI related questions",
    model=model,
    handoff_description="OpenAI Expert"
)
Agentic_AI : Agent = Agent(
    name= "Agenti AI Agent",
    instructions="You are an Agentic AI agent, you answer Agentic AI Developmnent related questions. You can use tools to answer sepcialized queries for specific tools requests",
    model=model,
    handoff_description="Agentic AI Expert",
    # handoffs=[DevOps, OpenAIAgent]
    tools=[
        DevOps.as_tool(
            tool_name="DevOps_tool",
            tool_description="Answer user's questions regarding Devops",
        ),
        OpenAIAgent.as_tool(
            tool_name="OpenAIAgent_tool",
            tool_description="Answer User's questions regarding OpenAI Agents",
        ),
        MobileDev.as_tool(tool_name="MobileDeveloper_tool", 
                          tool_description="Answer Mobile Development related queries"),

        WebDev.as_tool(tool_name="WebsiteDevelopent_tool",
                        tool_description="Answer Web Development related queries"),

    ],
)

# def on_web_handoff(context):
#     print("Web Agent Called")

# def on_DevOps_handoff(context):
#     print("DevOps Agent Called")

# def on_mobile_handoff(context):
#     print("Mobile Dev Called")

# def on_agentic_handoff(context):
#     print("Agentic AI Called")


Panacloud = Agent(name="Assistant",
                  instructions="""
    You are a routing agent that determines which specialized agent should handle the user's request based on the inquiry's content.
    Analyze the inquiry and route it to the appropriate agent using the following rules:
    - If the inquiry contains terms like 'web', 'API', 'HTML', 'CSS', 'JavaScript', 'backend', or 'frontend', route to the WebDev agent.
    - If the inquiry contains terms like 'mobile', 'app', 'iOS', 'Android', 'Flutter', or 'React Native', route to the MobileDev agent.
    - If the inquiry contains terms like 'agentic', 'AI agent', 'autonomous', or 'multi-agent', route to the Agentic_AI agent.
    - If the inquiry is unclear, ask the user for clarification.
    Only route to one agent. Do not trigger multiple agents.
    """,
                  model=model,)
                #   handoffs=[handoff(WebDev,on_handoff=on_web_handoff),
                #             handoff(MobileDev,on_handoff=on_mobile_handoff),
                #             handoff(Agentic_AI,on_handoff=on_agentic_handoff),
                #                     ])




