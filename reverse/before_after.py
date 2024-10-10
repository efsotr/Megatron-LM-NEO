import torch
def reverse_input_except_padding(tokenized_prompts, tokenizer, padding_token=None, keep_end_bos=False, must_add_begin_eos=True):
    # Set padding token if not provided
    if padding_token is None:
        padding_token = tokenizer.pad_token_id

    # Extract input_ids and attention_mask
    input_ids = tokenized_prompts['input_ids']
    attention_mask = tokenized_prompts['attention_mask']
    
    # Check if input is batched or single tensor
    was_single_tensor = False
    if input_ids.ndim == 1:
        was_single_tensor = True
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

    # Initialize new tensors to hold the possibly expanded sequences
    new_length = input_ids.size(1) + int(must_add_begin_eos and input_ids[0][-1] != tokenizer.eos_token_id)  # Additional space for EOS if required
    new_input_ids = torch.full((input_ids.size(0), new_length), padding_token, dtype=input_ids.dtype, device=input_ids.device)
    new_attention_mask = torch.zeros((input_ids.size(0), new_length), dtype=attention_mask.dtype, device=attention_mask.device)
    # breakpoint()
    # Reverse the sequences except for padding
    for i in range(input_ids.size(0)):
        # Identify the actual start and end of the non-padding tokens
        valid_indices = attention_mask[i].nonzero().squeeze()
        if valid_indices.numel() == 0:  # Handle edge case where the entire sequence might be padding
            continue

        start, end = valid_indices[0], valid_indices[-1] + 1
        # Reverse the sequence within the bounds of valid tokens
        reversed_ids = input_ids[i, start:end].flip(dims=[0])
        reversed_mask = attention_mask[i, start:end].flip(dims=[0])

        # Handle end BOS token if present
        if not keep_end_bos and reversed_ids[-1] == tokenizer.bos_token_id:
            reversed_ids = reversed_ids[:-1]
            reversed_mask = reversed_mask[:-1]
            end -= 1

        # Set up the new indices for copying
        new_start = start+1 if must_add_begin_eos and reversed_ids[0] != tokenizer.eos_token_id else start
        new_end = new_start + (end - start)

        # Insert EOS at the beginning if required
        if must_add_begin_eos and reversed_ids[0] != tokenizer.eos_token_id:
            new_input_ids[i, new_start-1] = tokenizer.eos_token_id
            new_attention_mask[i, new_start-1] = 1

        # Copy the reversed sequence into the new tensor
        new_input_ids[i, new_start:new_end] = reversed_ids
        new_attention_mask[i, new_start:new_end] = reversed_mask

    # if the last token is padding, we need to remove it
    if new_input_ids[:, -1].eq(padding_token).all():
        new_input_ids = new_input_ids[:, :-1]
        new_attention_mask = new_attention_mask[:, :-1]
    # If it was initially a single tensor, remove the batch dimension
    if was_single_tensor:
        new_input_ids = new_input_ids.squeeze(0)
        new_attention_mask = new_attention_mask.squeeze(0)

    # Return the updated dictionary containing reversed input_ids and attention_mask
    from types import SimpleNamespace
    return SimpleNamespace(input_ids=new_input_ids, attention_mask=new_attention_mask)

def reverse_output_ids(output_ids):
    # Check if the output is batched or single tensor
    was_single_tensor = False
    if output_ids.ndim == 1:
        was_single_tensor = True
        output_ids = output_ids.unsqueeze(0)  # Add a batch dimension for uniformity
    # Reverse the output IDs
    reversed_output_ids = output_ids.flip(dims=[1])  # Flip along the sequence length dimension
    # If it was initially a single tensor, remove the batch dimension before returning
    if was_single_tensor:
        reversed_output_ids = reversed_output_ids.squeeze(0)
    return reversed_output_ids
