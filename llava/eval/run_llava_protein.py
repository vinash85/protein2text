import argparse
import torch
import re
from llava.constants_protein import (
    PROTEIN_SEQUENCE_TOKEN_INDEX,
    DEFAULT_PROTEIN_SEQUENCE_TOKEN,
    DEFAULT_PROT_START_TOKEN,
    DEFAULT_PROT_END_TOKEN,
    PROTEIN_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder_protein import load_pretrained_model
from llava.utils import disable_torch_init
from llava.train.train_protein import protein_sequence_tokenizer
from llava.mm_utils import get_model_name_from_path

def inference_protein(model, tokenizer, prompt, amino_seq,
                       temperature=0.2, top_p=None, num_beams=1, max_new_tokens=512):
    # Tokenize the input prompt using the protein sequence tokenizer
    input_ids = (
        protein_sequence_tokenizer(prompt, tokenizer, PROTEIN_SEQUENCE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # Generate the model output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            amino_seq,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    # Decode the output tokens and print the result
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def eval_model(args):
    # Disable unnecessary torch initialization
    disable_torch_init()

    # Load the model and tokenizer
    print(f"Loading model: {args.model_path}")
    model_name = get_model_name_from_path(args.model_path)

    print(f"Loading model: {model_name}")

    tokenizer, model, _, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # Prepare the input query
    qs = args.query
    amino_seq = args.amino_seq

    protein_token_se = DEFAULT_PROT_START_TOKEN + DEFAULT_PROTEIN_SEQUENCE_TOKEN + DEFAULT_PROT_END_TOKEN

    # Replace the placeholder with the appropriate protein token structure
    if PROTEIN_PLACEHOLDER in qs:
        if model.config.mm_use_protein_start_end:
            qs = re.sub(PROTEIN_PLACEHOLDER, protein_token_se, qs)
        else:
            qs = re.sub(PROTEIN_PLACEHOLDER, DEFAULT_PROTEIN_SEQUENCE_TOKEN, qs)
    else:
        if model.config.mm_use_protein_start_end:
            qs = protein_token_se + "\n" + qs
        else:
            qs = DEFAULT_PROTEIN_SEQUENCE_TOKEN + "\n" + qs

    # Determine the conversation mode based on the model
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] The auto-inferred conversation mode is {}, but `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # Prepare the conversation template
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the input prompt using the protein sequence tokenizer
    input_ids = (
        protein_sequence_tokenizer(prompt, tokenizer, PROTEIN_SEQUENCE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # Generate the model output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            amino_seq,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    # Decode the output tokens and print the result
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--query", type=str, required=True)  # Direct query string input (protein sequences)
    parser.add_argument("--amino_seq", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
