import argparse
import datetime
import json
import os
import time
import hashlib
import gradio as gr
import requests

from llava.conversation_protein import default_conversation, conv_templates, SeparatorStyle
from llava.constants_protein import LOGDIR, VALID_AMINO_ACIDS
from llava.utils import build_logger, server_error_msg

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaVA Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models

get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
}
"""

def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update

def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update

def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3

def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3

def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3

def regenerate(state, protein_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

def is_valid_amino_acid_sequence(sequence):
    """
    Check if the given sequence only contains valid amino acids.
    """
    sequence = sequence.upper()
    return all(char in VALID_AMINO_ACIDS for char in sequence)

def validate_protein_sequence(protein_sequence):
    """
    Validate the uploaded or entered protein sequence and return an error if invalid.
    """
    if not is_valid_amino_acid_sequence(protein_sequence):
        return None, "Invalid protein sequence. Please enter a valid sequence containing only amino acid letters (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)."
    return protein_sequence, None

def process_input(state, textbox, protein_file_upload, protein_process_mode, request: gr.Request):
    """
    Process the input from the file upload (binary), textbox, and validate the protein sequence.
    """
    protein_sequence = None
    logger.info(f"Processing input. Text length: {len(textbox)}")

    # Check if `state` is a Conversation object and if we can add sequences to it
    if not hasattr(state, "sequences"):
        state.sequences = []  # Initialize if it doesn't exist

    # If a file is uploaded, read and validate the sequence from binary data
    if protein_file_upload:
        try:
            # Decode the binary content into a string assuming it is a text file
            file_content = protein_file_upload.decode("utf-8").strip()

            # Validate the uploaded protein sequence
            protein_sequence, error_msg = validate_protein_sequence(file_content)

            if error_msg:
                # If there is a validation error, return an error message
                return state, [["System", error_msg]], None, gr.update(visible=True), disable_btn, disable_btn, disable_btn, disable_btn, disable_btn

            # Add the validated protein sequence to the state object
            state.sequences.append(protein_sequence)

        except Exception as e:
            # Handle file reading errors
            return state, [["System", f"Error reading the uploaded file: {e}"]], None, gr.update(visible=True), disable_btn, disable_btn, disable_btn, disable_btn, disable_btn

    # Process the text input from the textbox
    if len(textbox) > 0:
        text = textbox[:1536]  # Hard cut-off for text
        if protein_sequence is not None:
            text = text[:1200]  # Cut-off for sequence input
            # if '<protein_sequence>' not in text:
                # text = text + '\n<protein_sequence>'  # Append sequence tag
            text = (text, protein_sequence, protein_process_mode)
        state.append_message(state.roles[0], text)
        state.append_message(state.roles[1], None)
        state.skip_next = False

    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        template_name = "vicuna_v1"  # Adjust template as needed
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address", json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()

    # Handle protein sequences similar to image handling
    all_sequences = state.get_sequences()  # Assuming you have a function to get sequences
    # all_sequence_hash = [hashlib.md5(sequence.encode()).hexdigest() for sequence in all_sequences]
    # for sequence, hash in zip(all_sequences, all_sequence_hash):
    #     t = datetime.datetime.now()
    #     filename = os.path.join(LOGDIR, "serve_sequences", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.txt")
    #     if not os.path.isfile(filename):
    #         os.makedirs(os.path.dirname(filename), exist_ok=True)
    #         with open(filename, "w") as f:
    #             f.write(sequence)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "sequences": f'List of {len(state.get_sequences())} sequences: {all_sequences}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['sequences'] = state.get_sequences()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:

        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream", headers=headers, json=pload, stream=True, timeout=120)
        
        logger.info(f"==== response status ==== {response.status_code}")

        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException: {e}")
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "sequences": all_sequences,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

title_markdown = ("""
# 🌋 LLaVul: Large Language and Code Vulnerability Assistant
[[Project Page](https://llava-vl.github.io)] [[Code](https://github.com/haotian-liu/LLaVA)] [[Model](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)] | 📚 [[LLaVA](https://arxiv.org/abs/2304.08485)] [[LLaVA-v1.5](https://arxiv.org/abs/2310.03744)] [[LLaVA-v1.6](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/)]
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")

learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""

def display_protein_status():
    return gr.update(value=f"<span style='color:green;'>File uploaded successfully.</span>", visible=True)


def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    # Start with a basic layout for the demo
    with gr.Blocks(title="LLaVul", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        # Column 1: Model selection and file upload
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        label="Select a model",
                        interactive=True,
                        show_label=True,
                        container=True
                    )

                # Keep only the file upload component for protein sequences
                status_message = gr.HTML(visible=True, label="File Status", value=" No file uploaded yet.")
                protein_file_upload = gr.File(
                    label="Upload a .txt file with a protein sequence", 
                    type="binary"
                )

                protein_process_mode = gr.Radio(
                    ["Default", "Custom"], value="Default", label="Preprocess for protein sequence", visible=False
                )

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P")
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens")

            # Column 2: Chatbot and inputs
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="LLaVul Chatbot",
                    height=650,
                    layout="panel"
                )

                # Textbox for asking questions about the protein, placeholder updated
                textbox = gr.Textbox(
                    label="Ask a question about the protein",
                    placeholder="Enter your prompt here"
                )

                # Submit button for sending input to the chatbot
                submit_btn = gr.Button(value="Send", variant="primary")

                # Control buttons
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        # Initialize the list of buttons for later use
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]

        # Terms and conditions
        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)

        url_params = gr.JSON(visible=False)


        # Use display_protein_status to update the label, then process_input
        protein_file_upload.change(
            fn=display_protein_status,  # Function to display status message
            inputs=[],  # Input for display_protein_status
            outputs=[status_message]  # Output to update the status message
            ).then(
            fn=process_input,  # Trigger process_input afterwards
            inputs=[state, textbox, protein_file_upload, protein_process_mode],  # Inputs for process_input
            outputs=[
                state,
                chatbot,  # Outputs updated chatbot and UI
                textbox,
                *btn_list
            ]
        )
        
        status_message.visible = False

        # Register listeners
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )

        regenerate_btn.click(
            regenerate,
            [state, protein_process_mode],
            [state, chatbot, textbox] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox] + btn_list,
            queue=False
        )

        # Submit button for input validation and chatbot interaction
        submit_btn.click(
            fn=process_input,
            inputs=[state, textbox, protein_file_upload, protein_process_mode],  # Removed protein_sequence_box
            outputs=[
                state,
                chatbot,
                textbox,
                *btn_list
            ]
        ).then(
            fn=http_bot,  # Call another function after the first one
            inputs=[state, model_selector, temperature, top_p, max_output_tokens],
            outputs=[state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        # Load initial demo configuration
        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
