# try:
#     from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
#     from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
#     from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
# except:
#     pass


try:
    from .language_model.llava_llama_protein import LlavaLlamaForCausalLM, LlavaConfig
except ImportError as e:
    print(f"ERROR: {e}")
    raise
# ['<image>red fluffy cat with a white face sleeps on a green mat with a soft paw and is looking at the\n']['<image>the 2016 - 2017 dubai festival map hall 13 stand e12\n']
# ['<image>whole cinnamon slices 500g\n']