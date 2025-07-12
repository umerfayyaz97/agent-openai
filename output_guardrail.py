from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    set_default_openai_api,set_default_openai_client,set_tracing_disabled,
    output_guardrail,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    OutputGuardrailTripwireTriggered,
    
)
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import sys

#for tracing locally
from agents import enable_verbose_stdout_logging
enable_verbose_stdout_logging()


load_dotenv()

set_tracing_disabled(True)

gemini_api_key = os.getenv('GEMINI_API_KEY')

client = AsyncOpenAI(api_key=gemini_api_key,
                     base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


set_default_openai_api("chat_completions")
set_default_openai_client(client)
model = "gemini-2.0-flash"


class MessageOutput(BaseModel):
    response: str

class MathOutput(BaseModel):
    is_math: bool
    reasoning: str

guardrail_agent2 = Agent(
    name="Guardrail check",
    instructions="Check if the output includes any math.",
    model=model,
    output_type=MathOutput,
)

@output_guardrail
async def math_guardrail2(
    wrapper: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent2, output.response, context=wrapper.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math,
    )

agent = Agent(
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    model=model,
    output_guardrails=[math_guardrail2],
    output_type=MessageOutput,
)

while True:
    query = input("Enter the query: ")
    if query == "quit":
        sys.exit()
    else:
        try:
            result = Runner.run_sync(
                agent,
                query,
            )
            print(result.final_output)
        except OutputGuardrailTripwireTriggered:
            print("Math output guardrail tripped")