from components.assets import arrow_circle_icon
from components.chat import chat, chat_form, chat_message, chat_messages
import asyncio
import modal
from fasthtml.common import *
import fastapi
import requests

MODELS_DIR = "/llamas"
MODEL_NAME = "ReliableAI/UCCIX-Llama2-13B"

# Download the model weights
try:
    volume = modal.Volume.lookup("llamas", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with the appropriate script")

# Define the Modal image with required dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "vllm==0.5.3post1",
    "python-fasthtml==0.4.3",
    "requests"
)

# Define the Modal app
app = modal.App("llama-chatbot")

# vLLM server implementation
@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    container_idle_timeout=5 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
    volumes={MODELS_DIR: volume},
)
@modal.asgi_app()
def serve_vllm():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.logger import RequestLogger
    import fastapi
    from fastapi.responses import StreamingResponse, JSONResponse
    import uuid
    import asyncio
    from typing import Optional, AsyncGenerator

    # Create a FastAPI app
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com",
        version="0.0.1",
        docs_url="/docs",
    )

    # Create an `AsyncLLMEngine`, the core of the vLLM server.
    engine_args = AsyncEngineArgs(
        model=MODELS_DIR + "/" + MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Get model config using the robust event loop handling
    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    # Initialize OpenAIServingChat
    request_logger = RequestLogger(max_log_len=256)  # Adjust max_log_len as needed
    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        [MODEL_NAME],  # served_model_names
        "assistant",   # response_role
        lora_modules=None,  # Adjust if you're using LoRA
        prompt_adapters=None,  # Adjust if you're using prompt adapters
        request_logger=request_logger,
        chat_template=None,  # Adjust if you have a specific chat template
    )


    @web_app.get("/v1/completions")
    async def get_completions(prompt: str, max_tokens: int = 100, stream: bool = False):
        request_id = str(uuid.uuid4())
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            stop=["Human:", "\n\n"]
        )
        
        system_prompt = "You are a helpful Irish English translator and tutor. Respond concisely and stay on topic."
        full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
        
        async def completion_generator() -> AsyncGenerator[str, None]:
            try:
                full_response = ""
                async for result in engine.generate(full_prompt, sampling_params, request_id):
                    if len(result.outputs) > 0:
                        new_text = result.outputs[0].text
                        if not full_response:  # This is the first chunk
                            # Remove any part of the system prompt or "Assistant:" that might be generated
                            new_text = new_text.split("Assistant:")[-1].lstrip()
                        
                        # Only yield the new part of the text
                        new_part = new_text[len(full_response):]
                        full_response = new_text
                        
                        if new_part:
                            yield new_part
                        
                        if full_response.strip().endswith((".", "!", "?")):
                            break  # Stop if we have a complete sentence
            except Exception as e:
                yield str(e)

        if stream:
            return StreamingResponse(completion_generator(), media_type="text/event-stream")
        else:
            completion = ""
            async for chunk in completion_generator():
                completion += chunk
            return JSONResponse(content={"choices": [{"text": completion.strip()}]})

    return web_app

# FastHTML web interface implementation
@app.function(
    image=image,
)
@modal.asgi_app()
def serve_fasthtml():
    fasthtml_app, rt = fast_app(ws_hdr=True)

    @rt("/")
    async def get():
        return Div(
            H1("Chat with Irish English translator and tutor bot"),
            chat(),
            cls="flex flex-col items-center min-h-screen bg-red-100",
        )

    @fasthtml_app.ws("/ws")
    async def ws(msg: str, send):
        chat_messages.append({"role": "user", "content": msg})
        await send(chat_form(disabled=True))
        await send(Div(chat_message(len(chat_messages) - 1), id="messages", hx_swap_oob="beforeend"))

        vllm_url = f"https://c123ian--llama-chatbot-serve-vllm.modal.run/v1/completions"
        response = requests.get(vllm_url, params={"prompt": msg, "max_tokens": 100, "stream": True}, stream=True)

        if response.status_code == 200:
            chat_messages.append({"role": "assistant", "content": ""})
            message_index = len(chat_messages) - 1
            await send(Div(chat_message(message_index), id="messages", hx_swap_oob="beforeend"))
            
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    text = chunk.decode('utf-8')
                    chat_messages[message_index]["content"] += text
                    await send(Span(text, id=f"msg-content-{message_index}", hx_swap_oob="beforeend"))
        else:
            message = "Error: Unable to get response from LLM."
            chat_messages.append({"role": "assistant", "content": message})
            await send(Div(chat_message(len(chat_messages) - 1), id="messages", hx_swap_oob="beforeend"))

        await send(chat_form(disabled=False))

    return fasthtml_app


if __name__ == "__main__":
    serve_vllm()  # Serve the vLLM server
    serve_fasthtml()  # Serve the FastHTML web interface


