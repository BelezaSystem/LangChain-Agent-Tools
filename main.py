from langchain_core.messages import HumanMessage
from langchain.tools import BaseTool
from rich import print

from tools import MathTools, IdentityTools
from agent import build_llm, build_messages, run_agent


def main(human_text: str) -> None:
    llm = build_llm()
    messages = build_messages(human_text)
    tools: list[BaseTool] = MathTools().as_tools() + IdentityTools().as_tools()
    messages = run_agent(llm, tools, messages)
    print(messages)


def chat() -> None:
    llm = build_llm()
    tools: list[BaseTool] = MathTools().as_tools() + IdentityTools().as_tools()
    messages = None
    print("Chat iniciado. Digite 'sair' para encerrar.")
    while True:
        try:
            user = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando chat.")
            break
        if not user:
            continue
        if user.lower() in ("sair", "exit", "quit"):
            print("Encerrando chat.")
            break
        if messages is None:
            messages = build_messages(user)
        else:
            messages.append(HumanMessage(user))
        messages = run_agent(llm, tools, messages)
        assistant_msg = messages[-1]
        print(f"Agente: {getattr(assistant_msg, 'content', assistant_msg)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Executa o agente com tools")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="Oi, sou Otacílio Beleza. Pode me falar quanto é 1.13 vezes 2.31?",
        help="Mensagem do usuário para o agente",
    )
    parser.add_argument(
        "-c",
        "--chat",
        action="store_true",
        help="Inicia um chat interativo com o agente",
    )
    args = parser.parse_args()
    if args.chat:
        chat()
    else:
        main(args.prompt)