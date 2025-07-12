from agents import (Agent,set_default_openai_api, set_default_openai_client,set_tracing_disabled , enable_verbose_stdout_logging, GuardrailFunctionOutput, RunContextWrapper, Runner, TResponseInputItem, input_guardrail, output_guardrail, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered)
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AsyncOpenAI
import sys

load_dotenv()

set_tracing_disabled(True)

gemini_api_key = os.getenv('GEMINI_API_KEY')

client = AsyncOpenAI(api_key=gemini_api_key,
                     base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


set_default_openai_api("chat_completions")
set_default_openai_client(client)
model = "gemini-2.0-flash"

class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str


guardrail_agent = Agent(
    name = "Guardrail Check",
    instructions="Check if the user is asking to do their math homework",
    output_type=MathHomeworkOutput,
    model = model
)

@input_guardrail
async def math_guardrail(
    wrapper: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(
        guardrail_agent, input, context=wrapper.context
    )

    return GuardrailFunctionOutput(
        output_info= result.final_output,
        tripwire_triggered= result.final_output.is_math_homework
    )


agent = Agent(
    name = "Customer Support Agent",
    instructions="You are a customer support agent, You help customers with their questions",
    model = model,
    input_guardrails=[math_guardrail]
)

while True:
    query = input("Enter your query:  ")

    if query == "quit":
        sys.exit()
    else:
        try:
            result = Runner.run_sync(agent, query)
            print(result.final_output)

        except InputGuardrailTripwireTriggered:
            print("Math Homework Query detected")