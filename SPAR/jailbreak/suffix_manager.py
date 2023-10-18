import torch

class SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
    
    def get_prompt(self, adv_string=None):
        if adv_string is not None:
            self.adv_string = adv_string

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction} {self.adv_string}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()
        encoding = self.tokenizer(prompt, add_special_tokens=False)
        toks = encoding.input_ids

        self.conv_template.messages = []

        self.conv_template.append_message(self.conv_template.roles[0], None)
        toks = self.tokenizer(self.conv_template.get_prompt(), add_special_tokens=False).input_ids
        self._start_token_slice = slice(0, 1)
        self._user_role_slice = slice(self._start_token_slice.stop, len(toks)+1)

        self.conv_template.update_last_message(f"{self.instruction}")
        toks = self.tokenizer(self.conv_template.get_prompt(), add_special_tokens=False).input_ids
        self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

        self.conv_template.update_last_message(f"{self.instruction} {self.adv_string}")
        toks = self.tokenizer(self.conv_template.get_prompt(), add_special_tokens=False).input_ids
        self._control_slice = slice(self._goal_slice.stop, len(toks))

        self.conv_template.append_message(self.conv_template.roles[1], None)
        toks = self.tokenizer(self.conv_template.get_prompt(), add_special_tokens=False).input_ids
        self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

        self.conv_template.update_last_message(f"{self.target}")
        toks = self.tokenizer(self.conv_template.get_prompt(), add_special_tokens=False).input_ids
        self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2)
        self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)

        self.conv_template.messages = []

        return prompt
    
    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt, add_special_tokens=False).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids