from typing import List
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import ValidationError

load_dotenv()


def build_llm():
    """Cria e retorna o LLM configurado."""
    return init_chat_model("openai:gpt-4o-mini")


def build_messages(human_text: str) -> List[BaseMessage]:
    """Cria o histórico inicial de mensagens."""
    system_message = SystemMessage(
        "You are a helpful assistant. You have access to tools. When the user asks "
        "for something, first look if you have a tool that solves that problem."
    )
    human_message = HumanMessage(human_text)
    return [system_message, human_message]


def run_agent(llm, tools: List[BaseTool], messages: List[BaseMessage]) -> List[BaseMessage]:
    """Executa o fluxo do agente com suporte a ferramentas."""
    llm_with_tools = llm.bind_tools(tools)
    llm_response = llm_with_tools.invoke(messages)
    messages.append(llm_response)

    if isinstance(llm_response, AIMessage) and getattr(llm_response, "tool_calls", None):
        call = llm_response.tool_calls[-1]
        name, args, id_ = call["name"], call["args"], call["id"]

        tools_by_name = {tool.name: tool for tool in tools}
        try:
            content = tools_by_name[name].invoke(args)
            status = "success"
        except (KeyError, IndexError, TypeError, ValidationError, ValueError) as error:
            content = f"Please, fix your mistakes: {error}"
            status = "error"

        tool_message = ToolMessage(content=content, tool_call_id=id_, status=status)
        messages.append(tool_message)

        llm_response = llm_with_tools.invoke(messages)
        messages.append(llm_response)

    return messages