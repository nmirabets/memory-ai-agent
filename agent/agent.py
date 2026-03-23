import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from agent.tools import save_memory
from agent.tools import TOOLS
import streamlit as st


load_dotenv()

MODEL = "grok-3-fast"

def agent(messages):

    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )

    completion = client.chat.completions.create(
        model=MODEL,
        tools=TOOLS,
        messages=messages
    )

    response = completion.choices[0].message

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_call_arguments = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "save_memory":
                response = save_memory(tool_call_arguments["memories"])

                # Uncomment this to see the memory tool calls
                messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
                
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )
                response = completion.choices[0].message

                return response.content

    else:
        return response.content