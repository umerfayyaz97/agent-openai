from agents import Agent, Runner, set_tracing_disabled, set_default_openai_client, set_default_openai_api, function_tool, handoff
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

# Web Development Agent
WebDev: Agent = Agent(
    name="WebDev Agent",
    instructions="""You are a Web Development Expert. You can help with anything related to web development, including but not limited to HTML, CSS, JavaScript, backend frameworks, frontend frameworks, API integration, and website deployment. You are familiar with both frontend and backend technologies.""",
    model=model,
    # handoff_description="Web Development Expert"
)

# Mobile Development Agent
MobileDev: Agent = Agent(
    name="MobileDev Agent",
    instructions="""You are a Mobile App Development Expert. You specialize in mobile app development for both iOS and Android. You are knowledgeable in mobile technologies such as Flutter, React Native, Swift, Kotlin, and can assist with mobile development, debugging, and deployment.""",
    model=model,
    # handoff_description="Mobile App Development Expert"
)

# DevOps Agent
DevOps: Agent = Agent(
    name="DevOps Agent",
    instructions="""You are a DevOps Expert. You help automate and streamline software development processes, including continuous integration (CI), continuous delivery (CD), cloud infrastructure management, containerization (Docker, Kubernetes), and monitoring tools. You assist in optimizing deployment pipelines and managing scalable systems.""",
    model=model,
    # handoff_description="DevOps Expert"
)

# OpenAI Agent
OpenAIAgent: Agent = Agent(
    name="OpenAI Agent",
    instructions="""You are an OpenAI Expert. You specialize in answering questions about OpenAI's models, APIs, and general usage. Whether it's about GPT models, embeddings, fine-tuning, or API usage, you can provide clear answers and solutions related to OpenAI technologies.""",
    model=model,
    # handoff_description="OpenAI Expert"
)

# Agentic AI Agent (Main routing agent)

Agentic_AI: Agent = Agent(
    name="Agentic AI Agent",
    instructions="""You are an Agentic AI expert. You handle specialized queries in fields like Web Development, Mobile Development, DevOps, and OpenAI. When the user asks about these domains, you should route the query to the corresponding tool for further processing.""",
    model=model,
    handoff_description="Agentic AI Expert",
    tools=[
        WebDev.as_tool(
            tool_name="WebsiteDevelopment_tool",
            tool_description="Answer questions related to web development, including front-end, back-end, and web frameworks."
        ),
        MobileDev.as_tool(
            tool_name="MobileDeveloper_tool",
            tool_description="Answer questions related to mobile development, including app creation for iOS and Android."
        ),
        DevOps.as_tool(
            tool_name="DevOps_tool",
            tool_description="Answer questions regarding DevOps practices, CI/CD, and infrastructure management."
        ),
        OpenAIAgent.as_tool(
            tool_name="OpenAIAgent_tool",
            tool_description="Answer questions related to OpenAI's models and technologies."
        ),
    ]
)

# On Handoff Functions
def on_web_handoff(context):
    print("Web Development Agent called. Handling web-related query.")

def on_DevOps_handoff(context):
    print("DevOps Agent called. Handling DevOps-related query.")

def on_mobile_handoff(context):
    print("Mobile Development Agent called. Handling mobile-related query.")

def on_agentic_handoff(context):
    print("Agentic AI Agent called. Handling Agentic AI-related query.")


# Panacloud (Main routing agent)
Panacloud = Agent(
    name="Assistant",
    instructions="""You are a generalist agent capable of handling general questions. If the inquiry is within your scope, you should respond directly. However, if the inquiry falls into one of the specialized domains listed below, you should route it to the Agentic AI agent for further processing. Analyze the inquiry and route it to the correct agent based on the content of the request:

    - If the inquiry mentions terms like 'web', 'API', 'HTML', 'CSS', 'JavaScript', 'backend', 'frontend', 'Next.js', 'animations', etc., route to the Agentic AI agent (which will handle WebDev queries).
    - If the inquiry mentions terms like 'mobile', 'app', 'iOS', 'Android', 'Flutter', or 'React Native', route to the Agentic AI agent (which will handle MobileDev queries).
    - If the inquiry mentions terms like 'DevOps', 'CI/CD', 'Kubernetes', 'cloud', or 'infrastructure', route to the Agentic AI agent (which will handle DevOps queries).
    - If the inquiry mentions terms like 'OpenAI', 'GPT', 'AI agent', 'machine learning', or 'API', route to the Agentic AI agent (which will handle OpenAI-related queries).
    - If the inquiry is related to Agentic AI (like multi-agent systems or autonomous agents), route to the Agentic AI agent for further processing.

    - If the inquiry is unclear or you cannot determine the right agent, ask the user for clarification.

    Only route to one agent at a time, ensuring that the query is fully handled by the appropriate agent. If the inquiry is general in nature (unrelated to the above domains), feel free to respond directly yourself.
    """,
    model=model,
    handoffs=[handoff(Agentic_AI, on_handoff=on_agentic_handoff)]  # Only handoff to Agentic_AI for specialized queries
)


# Panacloud (Main routing agent)
# Panacloud = Agent(
#     name="Assistant",
#     instructions="""You are a generalist agent capable of handling general questions. If the inquiry is within your scope, you should respond directly. However, if the inquiry falls into one of the specialized domains listed below, you should route it to the appropriate specialized agent (Agentic AI) for further processing. Analyze the inquiry and route it to the correct agent based on the content of the request:
    
#     - If the inquiry mentions terms like 'web', 'API', 'HTML', 'CSS', 'JavaScript', 'backend', or 'frontend', route to the WebDev agent.
#     - If the inquiry mentions terms like 'mobile', 'app', 'iOS', 'Android', 'Flutter', or 'React Native', route to the MobileDev agent.
#     - If the inquiry mentions terms like 'DevOps', 'CI/CD', 'Kubernetes', 'cloud', or 'infrastructure', route to the DevOps agent.
#     - If the inquiry mentions terms like 'OpenAI', 'GPT', 'AI agent', 'machine learning', or 'API', route to the OpenAI agent.
#     - If the inquiry is related to Agentic AI (like multi-agent systems or autonomous agents), route to the Agentic AI agent.
    
#     - If the inquiry is unclear or you cannot determine the right agent, ask the user for clarification.
    
#     Only route to one agent at a time, ensuring that the query is fully handled by the appropriate agent. If the inquiry is general in nature, feel free to respond directly yourself.
#     """,
#     model=model,
#     handoffs=[handoff(WebDev, on_handoff=on_web_handoff),
#              handoff(MobileDev, on_handoff=on_mobile_handoff),
#              handoff(Agentic_AI, on_handoff=on_agentic_handoff),
#              handoff(DevOps, on_handoff=on_DevOps_handoff)]
# )

# Running the main query
query = input("Enter query: ")

result = Runner.run_sync(Panacloud, query)

# Print the final output and the agent called
print(f"Agent called: {result.last_agent.name}")
print(f"Final response: {result.final_output}")
