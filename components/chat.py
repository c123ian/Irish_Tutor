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
        cls="input input-bordered w-full max-w-xs"
    )

def chat_button(disabled=False):
    return Button(
        send_icon(),
        id="send-button",
        disabled=disabled,
        cls="btn btn-primary bg-red-200 hover:bg-red-600 text-white rounded-md p-2.5 flex items-center justify-center border border-red-500 focus:ring-2 focus:ring-red-300 disabled:bg-red-800 disabled:border-red-600 disabled:cursor-not-allowed"
    )

def chat_form(disabled=False):
    return Form(
        chat_input(disabled=disabled),
        chat_button(disabled=disabled),
        id="form",
        ws_send=True,
        cls="flex gap-2 items-center border-t border-base-300 p-2"
    )


def chat_message(msg_idx):
    # Render each chat message with dynamic styling based on role
    msg = chat_messages[msg_idx]
    is_user = msg['role'] == 'user'
    #content = "..." if msg['content'] == "" else msg['content']
    return Div(
        Div(msg["role"], cls="chat-header opacity-50"),
        Div(msg["content"], cls=f"chat-bubble chat-bubble-{'primary' if is_user else 'secondary'}", id=f"msg-content-{msg_idx}"),
        id=f"msg-{msg_idx}",
        cls=f"chat chat-{'end' if is_user else 'start'}"
    )

#def chat_message(msg_idx):
#    msg = chat_messages[msg_idx]
#    bubble_class = "chat-bubble-primary" if msg['role'] == 'user' else 'chat-bubble-secondary'
#    chat_class = "chat-end" if msg['role'] == 'user' else 'chat-start'
#    content = "..." if msg['content'] == "" else msg['content']
#    
#    return Div(
#        Div(msg["role"], cls="chat-header text-xs text-red-500 mb-1 font-celtic"),
#        Div(
#            content,
#            cls=f"chat-bubble {bubble_class} bg-{'red-600 text-white' if msg['role'] == 'user' else 'red-200 text-black'}",
#            id=f"msg-content-{msg_idx}",
#        ),
#        id=f"msg-{msg_idx}",
#        cls=f"chat {chat_class}"
#    )

def chat_window():
    return Div(
        id="messages",
        *[chat_message(i) for i in range(len(chat_messages))],
        cls="flex flex-col gap-2 p-4 h-[73vh] overflow-y-auto w-full"
    )

#def chat_title():
#    return H1(
#        "Chat with Irish English translator and tutor bot",
#        cls="text-2xl font-bold mb-4 text-red-600 font-celtic"
#    )
def chat():
    # Combines the chat window and form into a full chat UI
    return Div(
        Div("Ask me to translate", cls="text-xs font-mono absolute top-0 left-0 w-fit p-1 bg-red border-b border-red-500 rounded-tl-md rounded-br-md font-celtic"),
        Div(f"The first message may take a while to process as the model loads.", 
            id="initial-message", 
            cls="text-sm font-mono w-full p-2 bg-yellow-100 border-b border-yellow-300 hidden"),
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
        cls="flex flex-col w-full max-w-2xl border border-base-300 h-full rounded-box shadow-lg relative bg-base-100"
    )

# old
#def chat():
#    return Div(
#        chat_title(),
#        chat_window(),
#        chat_form(),
#        Script(
#            """
#            function scrollToBottom(smooth) {
#                var messages = document.getElementById('messages');
#                messages.scrollTo({
#                    top: messages.scrollHeight,
#                    behavior: smooth ? 'smooth' : 'auto'
#                });
#            }
#            window.onload = function() {
#                scrollToBottom(true);
#            };
#            
#            const observer = new MutationObserver(function() {
#                scrollToBottom(false);
#            });
#            observer.observe(document.getElementById('messages'), { childList: true, subtree: true });
#            """
#        ),
#        hx_ext="ws",
#        ws_connect="/ws",
#        cls="flex flex-col items-center max-w-2xl mx-auto border border-red-500 rounded-md shadow-lg bg-white font-celtic p-4"
#   )