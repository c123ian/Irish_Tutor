import asyncio
import modal
from fasthtml.common import *
import fastapi
import aiohttp
from typing import AsyncGenerator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security, HTTPException, Depends

# Constants for model directory, model name, and token for authentication
MODELS_DIR = "/llamas"
MODEL_NAME = "ReliableAI/UCCIX-Llama2-13B-Instruct"
TOKEN = "XXXXXX"  # Replace with a modal.Secret

# Uncomment for token check, ensure a valid token is set
# Uncomment to check if the secret is set: 
# if not secret.get("TOKEN", None):
#    print("WARNING: Security token is not set. Please set it using modal secrets.")
#    raise Exception("Security token not set")

# Ensure volume for models is available
try:
    volume = modal.Volume.lookup("llamas", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with the appropriate script")

# Configure the container image with necessary dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "vllm==0.5.3post1",  # LLM engine
    "python-fasthtml==0.6.2",  # FastHTML for web rendering
    "aiohttp",  # HTTP client library for async requests
    "fastapi",  # FastAPI for async API endpoints
    "uvicorn"  # ASGI server to serve FastAPI/FastHTML
)

app = modal.App("irish-chatbot")  # Initialize Modal app

# Define FastHTML app with headers for Tailwind and DaisyUI
fasthtml_app, rt = fast_app(
    hdrs=(
        Script(src="https://cdn.tailwindcss.com"),
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
    )
)

# Chat messages and counter to manage state
chat_messages = []
message_count = 0

# Helper functions to render the chat UI
def chat_input(disabled=False):
    return Input(
        type="text",
        name="msg",
        id="msg-input",
        placeholder="Type a message",
        disabled=disabled,
        cls="input input-bordered w-full max-w-xs"
    )

def chat_button(disabled=False):
    return Button(
        "Send",
        id="send-button",
        disabled=disabled,
        cls="btn btn-primary"
    )

def chat_form(disabled=False):
    return Form(
        chat_input(disabled),
        chat_button(disabled),
        id="form",
        hx_post="/chat",  # HTMX form submission for live updates
        hx_target="#messages",  # Append new message to messages div
        hx_swap="beforeend",
        cls="flex gap-2 items-center border-t border-base-300 p-2"
    )

def chat_message(msg_idx):
    # Render each chat message with dynamic styling based on role
    msg = chat_messages[msg_idx]
    is_user = msg['role'] == 'user'
    return Div(
        Div(msg["role"], cls="chat-header opacity-50"),
        Div(msg["content"], cls=f"chat-bubble chat-bubble-{'primary' if is_user else 'secondary'}", id=f"msg-content-{msg_idx}"),
        id=f"msg-{msg_idx}",
        cls=f"chat chat-{'end' if is_user else 'start'}"
    )

def chat_window():
    # Renders the main chat window with scroll behavior for new messages
    return Div(
        id="messages",
        *[chat_message(i) for i in range(len(chat_messages))],
        cls="flex flex-col gap-2 p-4 h-[45vh] overflow-y-auto w-full",
        style="scroll-behavior: smooth;"
    )

def chat():
    # Combines the chat window and form into a full chat UI
    return Div(
        Div("Ask me to translate", cls="text-xs font-mono absolute top-0 left-0 w-fit p-1 bg-red border-b border-red-500 rounded-tl-md rounded-br-md font-celtic"),
        Div(f"The first message may take a while to process as the model {MODEL_NAME} loads.", 
            id="initial-message", 
            cls="text-sm font-mono w-full p-2 bg-yellow-100 border-b border-yellow-300 hidden"),
        chat_window(),
        chat_form(),
        cls="flex flex-col w-full max-w-2xl border border-base-300 h-full rounded-box shadow-lg relative bg-base-100"
    )

# FastAPI app to serve LLM completions
@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),  # Configures GPU resources
    container_idle_timeout=5 * 60,  # Timeout for idle containers
    timeout=24 * 60 * 60,  # Max function timeout (24 hours)
    allow_concurrent_inputs=100,  # Allow up to 100 concurrent requests
    volumes={MODELS_DIR: volume},  # Attach model volume
)
@modal.asgi_app()
def serve_vllm():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    import uuid
    import traceback

    web_app = fastapi.FastAPI()

    # Add CORS middleware for cross-origin requests
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    http_bearer = HTTPBearer(scheme_name="Bearer Token")

    # Authentication logic to validate bearer token
    async def is_authenticated(credentials: HTTPAuthorizationCredentials = Security(http_bearer)):
        if credentials.credentials != TOKEN:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
            )
        return {"username": "authenticated_user"}

    def find_model_path(base_dir, model_name):
        # Locate the model files based on directory structure
        for root, dirs, files in os.walk(base_dir):
            if "config.json" in files:
                return root
        return None

    # Check if model files exist
    model_path = find_model_path(MODELS_DIR, MODEL_NAME)
    if not model_path:
        raise Exception(f"Could not find model files for {MODEL_NAME} in {MODELS_DIR}")

    print(f"Initializing AsyncLLMEngine with model path: {model_path}")
    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=1,  # Single GPU
        gpu_memory_utilization=0.90,  # Max GPU utilization
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("AsyncLLMEngine initialized successfully")

    @web_app.get("/v1/completions")
    async def get_completions(prompt: str, max_tokens: int = 100, user=Depends(is_authenticated)):
        print(f"Received prompt: {prompt}")
        try:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                stop=["Human:", "\n\n"]
            )
            
            system_prompt = "You are a helpful Irish English translator and tutor. Respond concisely and stay on topic."
            full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"  # Construct full prompt
            request_id = str(uuid.uuid4())  # Unique ID for the request
            print(f"Generated request_id: {request_id}")
            
            # Async generator to stream LLM response in chunks (SSE-like approach)
            async def completion_generator():
                try:
                    full_response = ""
                    last_yielded_position = 0
                    assistant_prefix_removed = False
                    async for result in engine.generate(full_prompt, sampling_params, request_id):
                        if len(result.outputs) > 0:
                            new_text = result.outputs[0].text
                            
                            if not assistant_prefix_removed:
                                new_text = new_text.split("Assistant:")[-1].lstrip()  # Strip Assistant prefix
                                assistant_prefix_removed = True
                            
                            if len(new_text) > last_yielded_position:
                                new_part = new_text[last_yielded_position:]
                                yield new_part  # Stream new part of the response
                                last_yielded_position = len(new_text)
                            
                            full_response = new_text
                            
                            if full_response.strip().endswith((".", "!", "?")):
                                break  # Stop generation when complete response is detected
                except Exception as e:
                    print(f"Error in completion_generator: {e}")
                    print(traceback.format_exc())
                    yield f"Error: {str(e)}"

            completion = ""
            async for chunk in completion_generator():
                completion += chunk
            print(f"Generated completion: {completion}")
            return fastapi.responses.JSONResponse(content={"choices": [{"text": completion.strip()}]})
        except Exception as e:
            print(f"Error in get_completions: {e}")
            print(traceback.format_exc())
            return fastapi.responses.JSONResponse(
                status_code=500,
                content={"error": f"An error occurred while processing your request: {str(e)}"}
            )

    return web_app

# Separate FastHTML server for chat UI and interaction
@app.function(image=image)
@modal.asgi_app()
def serve_fasthtml():
    @rt("/")
    async def get():
        # Render the main chat interface with header
        return Div(
            H1("Chat with Irish English translator and tutor bot"),
            chat(),
            cls="flex flex-col items-center min-h-screen bg-black-100"
        )

    def message_preview(msg_idx):
        # Show preview while waiting for assistant response, using HTMX for polling
        if msg_idx < len(chat_messages) and chat_messages[msg_idx]['role'] == 'assistant':
            return chat_message(msg_idx)
        else:
            return Div(
                Div("assistant", cls="chat-header opacity-50"),
                Div(Span(cls="loading loading-dots loading-sm"), cls="chat-bubble chat-bubble-secondary"),
                id=f"msg-{msg_idx}",
                cls="chat chat-start",
                hx_get=f"/chat/{msg_idx}",  # HTMX poll every second, not efficent will change this
                hx_trigger="every 1s",
                hx_swap="outerHTML"
            )

    @rt("/chat")
    async def post(msg: str):
        global message_count
        message_count += 1
        
        chat_messages.append({"role": "user", "content": msg})  # Append user message
        user_message = chat_message(len(chat_messages) - 1)
        
        assistant_preview = message_preview(len(chat_messages))  # Show assistant loading preview
        
        clear_input = Input(id="msg-input", name="msg", placeholder="Type a message", hx_swap_oob="true")
        
        asyncio.create_task(generate_response(msg, len(chat_messages)))  # Async task to handle LLM response
        
        if message_count == 1:
            show_initial_message = Script("document.getElementById('initial-message').classList.remove('hidden'); setTimeout(() => document.getElementById('initial-message').classList.add('hidden'), 19000);")
            return user_message, assistant_preview, clear_input, show_initial_message
        else:
            return user_message, assistant_preview, clear_input

    @rt("/chat/{msg_idx}")
    async def get(msg_idx: int):
        return message_preview(msg_idx)  # Update preview for assistant message

    async def generate_response(msg: str, msg_idx: int):
        # Prepare prompt and send request to LLM server for response
        full_conversation = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_messages])
        prompt = f"{full_conversation}\nHuman: {msg}\nAssistant:"
        
        vllm_url = f"https://c123ian--irish-chatbot-serve-vllm.modal.run/v1/completions"
        headers = {"Authorization": f"Bearer {TOKEN}"}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(vllm_url, params={
                    "prompt": prompt, 
                    "max_tokens": 100
                }, headers=headers, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()  # Get LLM response
                        assistant_response = data['choices'][0]['text']
                        chat_messages.append({"role": "assistant", "content": assistant_response})
                    else:
                        error_message = f"Error: Unable to get response from LLM. Status: {response.status}"
                        chat_messages.append({"role": "assistant", "content": error_message})
            except aiohttp.ClientError as e:
                error_message = f"Error: Unable to connect to the LLM server. {str(e)}"
                chat_messages.append({"role": "assistant", "content": error_message})

    return fasthtml_app  # Return FastHTML app

if __name__ == "__main__":
    serve_vllm()  # Start LLM server
    serve_fasthtml()  # Start FastHTML server for chat interface
