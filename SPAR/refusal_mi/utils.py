from typing import List
from torch import Tensor
from jaxtyping import Int, Float
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def instruction_to_prompt(
    instruction: str,
    system_prompt: str=None,
    model_output: str=None,
    append_space: bool=True,
) -> str:
    """
    Converts an instruction to a prompt string structured for Llama2-chat.
    See details of Llama2-chat prompt structure here: here https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """
    if system_prompt is not None:
        dialog_content = B_SYS + system_prompt.strip() + E_SYS + instruction.strip()
    else:
        dialog_content = instruction.strip()

    if model_output is not None:
        prompt = f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
    else:
        prompt = f"{B_INST} {dialog_content.strip()} {E_INST}"

    if append_space:
        prompt = prompt + " "

    return prompt

def tokenize_instructions(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    padding_length: int=None,
    system_prompt: str=None,
    model_outputs: List[str]=None,
) -> Int[Tensor, "batch seq_len"]:
    if model_outputs is not None:
        assert(len(instructions) == len(model_outputs))
        prompts = [
            instruction_to_prompt(instruction, system_prompt, model_output, append_space=False)
            for (instruction, model_output) in zip(instructions, model_outputs)
        ]
    else:
        prompts = [
            instruction_to_prompt(instruction, system_prompt, model_output=None, append_space=True)
            for instruction in instructions
        ]

    instructions_toks = tokenizer(
        prompts,
        padding="max_length" if padding_length is not None else True,
        max_length=padding_length,
        truncation=False,
        return_tensors="pt"
    ).input_ids

    return instructions_toks

def generate_from_instructions(
    tl_model: HookedTransformer,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    max_new_tokens: int=64,
    temperature: float=0.0,
):
    instructions_toks = tokenize_instructions(tokenizer, instructions)

    output_ids = tl_model.generate(instructions_toks, max_new_tokens=max_new_tokens, temperature=temperature)
    for answer_idx, answer in enumerate(tokenizer.batch_decode(output_ids)):
        print(f"\nGeneration #{answer_idx}:\n\t{repr(answer)}")