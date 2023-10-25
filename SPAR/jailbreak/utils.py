import plotly.graph_objects as go
import torch

def get_nonascii_nonspecial_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()
    
    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)
    
    special_toks = {
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id
    }

    for special_tok in special_toks:
        nonascii_toks.append(special_tok)

    return torch.tensor(nonascii_toks, device=device)

def get_filtered_cand_toks(tokenizer, control_cand, curr_control=None):
    cand_toks = []
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
            cand_toks.append(control_cand[i])

    return torch.stack(cand_toks)
