# failed.
from transformers import AutoTokenizer
class ReversedTokenIDsTokenizer(AutoTokenizer):
    def _encode(self, text, **kwargs):
        # Call the original method to get the encoded output
        encoded = super()._encode(text, **kwargs)
        # Reverse the token IDs, handle padding and special tokens appropriately
        return self.modify_token_ids(encoded, kwargs.get('padding', False))

    def modify_token_ids(self, token_ids, is_padded):
        # Reverse the token IDs and handle padding tokens
        if is_padded:
            pad_token_id = self.pad_token_id
            non_padded_tokens = [tid for tid in token_ids if tid != pad_token_id]
            reversed_ids = list(reversed(non_padded_tokens))
            result_ids = []
            for tid in token_ids:
                if tid == pad_token_id:
                    result_ids.append(pad_token_id)
                else:
                    result_ids.append(reversed_ids.pop(0))
        else:
            reversed_ids = list(reversed(token_ids))
        
        # Modify for start and end tokens as needed
        if reversed_ids and reversed_ids[-1] == 1:
            reversed_ids.pop(-1)
        reversed_ids.insert(0, 2)
        return reversed_ids

    def decode(self, token_ids, **kwargs):
        if token_ids and token_ids[0] == 2:
            token_ids = token_ids[1:]
        return super().decode(list(reversed(token_ids)), **kwargs)




class ReversedTokenIDsTokenizer_old:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        outputs = self._tokenizer(*args, **kwargs)
        outputs['input_ids'] = [self.modify_token_ids(ids, kwargs.get('padding', False)) for ids in outputs['input_ids']]
        return outputs

    def encode(self, *args, **kwargs):
        ids = self._tokenizer.encode(*args, **kwargs)
        return self.modify_token_ids(ids, kwargs.get('padding', False))

    def encode_plus(self, *args, **kwargs):
        outputs = self._tokenizer.encode_plus(*args, **kwargs)
        outputs['input_ids'] = self.modify_token_ids(outputs['input_ids'], kwargs.get('padding', False))
        return outputs

    def batch_encode_plus(self, *args, **kwargs):
        outputs = self._tokenizer.batch_encode_plus(*args, **kwargs)
        outputs['input_ids'] = [self.modify_token_ids(ids, kwargs.get('padding', False)) for ids in outputs['input_ids']]
        return outputs

    def modify_token_ids(self, token_ids, is_padded):
        # Find out if padding is applied
        pad_token_id = self._tokenizer.pad_token_id
        # Reverse the token IDs while keeping padding tokens in place if padding is enabled
        if is_padded and pad_token_id is not None:
            non_padded_tokens = [tid for tid in token_ids if tid != pad_token_id]
            reversed_ids = list(reversed(non_padded_tokens))
            # Replace non-pad tokens with reversed ones
            result_ids = []
            for tid in token_ids:
                if tid == pad_token_id:
                    result_ids.append(pad_token_id)
                else:
                    result_ids.append(reversed_ids.pop(0))
        else:
            reversed_ids = list(reversed(token_ids))
        
        # Remove the ending token if it is 1
        if reversed_ids and reversed_ids[-1] == 1:
            reversed_ids.pop(-1)
        # Add 2 at the beginning
        reversed_ids.insert(0, 2)
        return reversed_ids

    def decode(self, token_ids, *args, **kwargs):
        # Reverse the order of token IDs before decoding, adjust for added 2 at start
        if token_ids and token_ids[0] == 2:
            token_ids = token_ids[1:]
        reversed_ids = list(reversed(token_ids))
        return self._tokenizer.decode(reversed_ids, *args, **kwargs)

    def batch_decode(self, batch_token_ids, *args, **kwargs):
        adjusted_batches = []
        for ids in batch_token_ids:
            if ids and ids[0] == 2:
                ids = ids[1:]
            reversed_ids = list(reversed(ids))
            adjusted_batches.append(reversed_ids)
        return self._tokenizer.batch_decode(adjusted_batches, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

def load_reverse_tokenizer(path):
    # Load the original tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(path)
    # Return the modified tokenizer
    return ReversedTokenIDsTokenizer.from_pretrained(path)

