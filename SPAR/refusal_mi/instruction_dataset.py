from typing import List
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import tokenize_instructions

DEFAULT_INSTRUCTION_TEMPLATE = "Tell me how to make a {object}."

class InstructionDataset(Dataset):
    def __init__(
        self,
        objects: List[str],
        tokenizer: AutoTokenizer,
        padding_length: int,
        instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    ):
        self.objects = objects
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.instruction_template = instruction_template

        self.prompt_toks = tokenize_instructions(
            tokenizer,
            instructions=[self.instruction_template.format(object=object) for object in objects],
            padding_length=self.padding_length
        )
        self.prompt_strs = [tokenizer.decode(self.prompt_toks[i]) for i in range(len(self.objects))]
        self.prompt_str_toks = [
            [tokenizer.decode(self.prompt_toks[i, j]) for j in range(padding_length)]
            for i in range(len(self.objects))
        ]

        self.object_tok_pos = self._get_last_object_tok_pos()

    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, idx):
        return self.prompt_toks[idx]

    def _get_last_object_tok_pos(self):
        single_tok_object = "pie"
        single_tok_object_toks = tokenize_instructions(
            self.tokenizer,
            instructions=[self.instruction_template.format(object=single_tok_object)],
            padding_length=self.padding_length
        )
        return [self.tokenizer.decode(tok) for tok in single_tok_object_toks[0]].index(single_tok_object)

class PairedInstructionDataset:
    def __init__(
        self,
        harmful_objects: List[str],
        harmless_objects: List[str],
        tokenizer: AutoTokenizer,
        prompt_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    ):
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template

        max_length = self._find_max_length(harmful_objects + harmless_objects, tokenizer, prompt_template)

        self.harmful_dataset = InstructionDataset(harmful_objects, tokenizer, max_length, prompt_template)
        self.harmless_dataset = InstructionDataset(harmless_objects, tokenizer, max_length, prompt_template)

    def _find_max_length(self, objects: List[str], tokenizer: AutoTokenizer, prompt_template: str):
        prompt_toks = tokenize_instructions(
            tokenizer,
            [prompt_template.format(object=object) for object in objects]
        )
        return prompt_toks.shape[1]