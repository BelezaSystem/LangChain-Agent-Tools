import gradio as gr
from langchain_core.messages import HumanMessage

from agent import build_llm, build_messages, run_agent
from tools import MathTools, IdentityTools


def submit_message(user_input: str, chat_history: list, messages):
    if not user_input:
        return chat_history, messages

    if messages is None:
        messages = build_messages(user_input)
    else:
        messages.append(HumanMessage(user_input))

    llm = build_llm()
    tools = MathTools().as_tools() + IdentityTools().as_tools()
    messages = run_agent(llm, tools, messages)

    assistant_msg = messages[-1]
    response_text = getattr(assistant_msg, "content", assistant_msg)
    chat_history = chat_history + [[user_input, response_text]]
    return chat_history, messages


def clear_chat():
    return [], None


def build_ui():
    with gr.Blocks(title="Chat do Agente") as demo:
        gr.Markdown("# Chat do Agente\nConverse com o agente com suporte a tools.")

        chatbot = gr.Chatbot(height=480)
        user = gr.Textbox(placeholder="Digite sua mensagem e pressione Enter", label="Mensagem")
        clear_btn = gr.Button("Limpar chat")

        messages_state = gr.State(None)

        user.submit(
            submit_message,
            inputs=[user, chatbot, messages_state],
            outputs=[chatbot, messages_state],
        )
        clear_btn.click(clear_chat, outputs=[chatbot, messages_state])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)