from fasthtml.common import *
from components.assets import send_icon

chat_messages = []


def chat_input(disabled=False):
    return Input(
        type="text",
        name="msg",
        id="msg-input",
        required=True,
        placeholder="Type a message",
        hx_swap_oob="true",
        autofocus="true",
        disabled=disabled,
        cls="!mb-0 bg-white border border-red-500 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-red-300 disabled:bg-gray-200 disabled:border-red-500 disabled:cursor-not-allowed rounded-md",
        style="width: 300px;"  # Setting a specific width, e.g., 300px
    )


def chat_button(disabled=False):
    return Button(
        send_icon(),
        id="send-button",
        disabled=disabled,
        cls="bg-red-500 hover:bg-red-600 text-white rounded-md p-2.5 flex items-center justify-center border border-red-500 focus-visible:outline-none focus-visible:ring-red-300 disabled:bg-red-800 disabled:border-red-600 disabled:cursor-not-allowed",
    )


def chat_form(disabled=False):
    return Form(
        chat_input(disabled=disabled),
        chat_button(disabled=disabled),
        id="form",
        ws_send=True,
        cls="w-full flex gap-2 items-center border-t border-red-500 p-2",
    )


def chat_message(msg_idx):
    msg = chat_messages[msg_idx]
    content_cls = f"px-2.5 py-1.5 rounded-lg max-w-xs {'rounded-br-none border-red-500 border' if msg['role'] == 'user' else 'rounded-bl-none border-red-300 border'}"

    return Div(
        Div(msg["role"], cls="text-xs text-red-500 mb-1 font-celtic"),
        Div(
            msg["content"],
            cls=f"bg-{'red-600 text-white' if msg['role'] == 'user' else 'red-200 text-black'} {content_cls}",
            id=f"msg-content-{msg_idx}",
        ),
        id=f"msg-{msg_idx}",
        cls=f"self-{'end' if msg['role'] == 'user' else 'start'}",
    )


def chat_window():
    return Div(
        id="messages",
        *[chat_message(i) for i in range(len(chat_messages))],
        cls="flex flex-col gap-2 p-4 h-[45vh] overflow-y-auto w-full",
    )


def chat_title():
    return Div(
        "Ask me to translate",
        cls="text-xs font-mono absolute top-0 left-0 w-fit p-1 bg-red border-b border-r border-red-500 rounded-tl-md rounded-br-md font-celtic",
    )


def chat():
    return Div(
        chat_title(),
        chat_window(),
        chat_form(),
        Script(
            """
            function scrollToBottom(smooth) {
                var messages = document.getElementById('messages');
                messages.scrollTo({
                    top: messages.scrollHeight,
                    behavior: smooth ? 'smooth' : 'auto'
                });
            }
            window.onload = function() {
                scrollToBottom(true);
            };

            const observer = new MutationObserver(function() {
                scrollToBottom(false);
            });
            observer.observe(document.getElementById('messages'), { childList: true, subtree: true });
            """
        ),
        hx_ext="ws",
        ws_connect="/ws",
        cls="flex flex-col w-full max-w-2xl border border-red-500 h-full rounded-md outline-1 outline outline-red-500 outline-offset-2 relative bg-white font-celtic",
    )

