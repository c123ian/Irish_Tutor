### Demo

I'll have it live for a little while (here)[https://c123ian--irish-chatbot-serve-fasthtml.modal.run/]


### Base Model

Using **UCCIX-Llama2-13B**, an Irish-English bilingual model based on **Llama 2-13B**. Capable of understanding both languages and outperforms larger models on Irish language tasks.

- Available at: https://huggingface.co/ReliableAI/UCCIX-Llama2-13B

Key aspects of the final code:

1. **Modal app configuration**: Define a Modal app and set up the image with dependencies.
2. **Volume setup**: Use a Modal volume to store model weights.
3. **vLLM server**: GPU-based language model inference, implemented with Modal.
4. **FastHTML interface**: Serve the web interface via FastHTML.
5. **Deployment**: Both vLLM server and FastHTML interface run as ASGI apps.

Two versions of the chatbot emerged, but for either chatbot you can insert your own llm from huggingface :) 

To deploy this application:

```bash
modal deploy irish_llm_v2.py
```
This command deploys both the vLLM server and FastHTML interface.

![image](https://github.com/user-attachments/assets/d9201394-cf3d-424b-9f84-5d9d5caf69a7)


Please note, if you would like to use your own model from Huggingface Hub, make sure to save teh weights to a Modal volume (in my case I call it `/llamas`).

```bash
modal run download_llama.py
```


---

### Chatbot Comparison

#### 1. **Irish Tutor LLM v1** (`irish_tutor_llm_v1_ws.py`)
- **Simplified code**:
  - Uses **websockets** (`@fasthtml_app.ws`) for real-time chat and non-blocking requests.
  - The **GUI is modularized**: chat components (e.g., `chat_form`, `chat_message`) are imported from external modules, keeping the main code clean and easier to maintain.
- **OpenAI `/v1/completions` API**: Standard OpenAI-compatible API for LLM responses.
- **Single prompt-response**: Handles individual exchanges without retaining conversation history.
- **Efficient for basic use**: No loading indicators or security features—straightforward for simple, fast setups.

#### 2. **Irish LLM v2** (`irish_llm_v2.py`)
- **More complex structure**:
  - Uses **`aiohttp`** for asynchronous HTTP requests (`/chat` routes), which handles multiple concurrent requests but adds complexity.
  - **In-code UI elements**: Chat form and message bubbles are embedded directly into the code (e.g., `chat_window`, `chat_message`), integrating the frontend with the main logic.
  - **Bearer token authentication** (`HTTPBearer`) ensures secure access to the API.
  - **Conversation history**: Appends previous messages into the prompt (`full_conversation`), enabling context-aware dialogue over multiple exchanges.
  - **Temporary loading indicator**: Implements a loading message (with HTMX and three dots) and a warning for CPU load times (`setTimeout` for showing the message).
- **Dynamic model loading**: Dynamically searches for the model configuration files (`find_model_path`), adding flexibility in deployment.
- **Improved response control**: Prevents overly long responses by managing token generation, ensuring clean completion and avoiding rambling (`completion_generator`).

---

### Notes:
- **`irish_tutor_llm_v1_ws.py`** (Irish Tutor LLM v1) is simpler, using **websockets** for interaction and modular UI components. It handles single prompt-response interactions, making it quick and easy to deploy for basic use cases.
- **`irish_llm_v2.py`** (Irish LLM v2) includes more advanced features like **security, conversation retention**, and **loading indicators**, but adds complexity due to **HTTP requests** and its focus on handling multi-turn conversations.
- This is essentially a combination of this template (which just echos back user input) https://github.com/arihanv/fasthtml-modal and Modal Labs [Run an OpenAI-Compatible vLLM Server](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/vllm_inference.py) tutorial. 
- Using OpenAI API's  "/v1/completions" rather then the more apporpriate "/v1/chat/completions", see where code was sourced [here]( https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1)
- The `irish_tutor_llm_v1_ws.py` uses FastHTML's websockest `/ws` rather then [FastHTML SSE](https://github.com/AnswerDotAI/fasthtml-example/blob/main/04_sse/sse_rand_scroll.py)
- The `irish_llm_v2.py` loading-dots animation uses polling, this sends a lot of http requests every 1 second to the server, so defentitly could be made more efficent!
- This uses UCCIX-Llama2-13B, you may need to request permission via Huggingface Hub and run the `download_llama.py` script in order to download weights onto a Modla labs Volume (which we call here `/llamas`).
- The code generate two URLs, one is the backend running on a Modal labs GPU, the second is the front-end (the FastHTML GUI running on a Modal Labs CPU) `https://USERNAME--llama-chatbot-serve-fasthtml.modal.run/`, I have yet to add some user affordance to alert user they have to wait for initial cold-boot response. 
