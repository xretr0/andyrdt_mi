import torch

from typing import List, Optional
from tokenize_llama import tokenize_llama_chat, E_INST

class SuffixManager:
    def __init__(self, *, tokenizer, instruction, target, adv_suffix):
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.target = target
        self.adv_suffix = adv_suffix
    
    def set_adv_suffix(self, adv_suffix):
        self.adv_suffix = adv_suffix

    def get_prompt(self):
        toks = self.get_input_ids()
        return self.tokenizer.decode(toks)

    def get_input_ids(self):
        toks = tokenize_llama_chat(
            self.tokenizer,
            conversation=[
                (
                    self.instruction + " " + self.adv_suffix,
                    self.target
                )
            ],
            no_final_eos=True,
        )

        self._update_slices(toks)

        return torch.tensor(toks)

    def _update_slices(self, toks):
        self._assistant_role_slice = get_sub_toks_slice(
           self.tokenizer.encode(E_INST, add_special_tokens=False),
           toks,
        )
        self._control_slice = slice(
            self._assistant_role_slice.start - len(self.tokenizer.encode(self.adv_suffix, add_special_tokens=False)),
            self._assistant_role_slice.start
        )
        self._target_slice = slice(
            self._assistant_role_slice.stop,
            len(toks)
        )
        self._loss_slice = slice(
            self._target_slice.start-1,
            self._target_slice.stop-1
        )

def get_sub_toks_slice(sub_toks: List[int], toks: List[int]) -> Optional[slice]:
    sub_len = len(sub_toks)
    toks_len = len(toks)
    
    for i in range(toks_len - sub_len + 1):
        if toks[i:i+sub_len] == sub_toks:
            return slice(i, i+sub_len)
    return None