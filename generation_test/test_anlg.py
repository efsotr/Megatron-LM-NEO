import json
import jsonlines
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

def load_jsonl(file_path, num_samples=None):
    with jsonlines.open(file_path, 'r') as reader:
        data = [item for item in reader]
    if num_samples:
        data = data[:num_samples]
    return data
def load_valid_examples(valid_jsonl_file, num_examples):
    return load_jsonl(valid_jsonl_file, num_examples)
def load_txt(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file.readlines()]
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def right_to_left_padding_batch(batch_input_ids, padding_token_id=1):
    if not isinstance(batch_input_ids, torch.Tensor):
        raise TypeError("batch_input_ids should be a torch.Tensor")
    
    # Get the device and dtype of the input tensor
    device = batch_input_ids.device
    dtype = batch_input_ids.dtype
    
    left_padded_batch = []
    
    # Convert each sequence to left padding
    for input_ids in batch_input_ids:
        input_ids = input_ids.tolist()  # Convert tensor to list for processing
        input_ids_no_padding = [token_id for token_id in input_ids if token_id != padding_token_id]
        num_padding_tokens = len(input_ids) - len(input_ids_no_padding)
        left_padded_input_ids = [padding_token_id] * num_padding_tokens + input_ids_no_padding
        left_padded_batch.append(left_padded_input_ids)
    
    # Convert the list of lists back to a tensor
    left_padded_batch_tensor = torch.tensor(left_padded_batch, dtype=dtype, device=device)
    
    return left_padded_batch_tensor



def generate_text(model, tokenizer, prompts, device, model_type, max_length=32, num_return_sequences=1, temperature=1.0, top_k=50, top_p=0.95):
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, return_token_type_ids=False).to(device)
    if model_type == 'blm':
        # Reverse the input token IDs for the backward language model
        inputs['input_ids'] = torch.flip(inputs['input_ids'], dims=[1])
        inputs['input_ids'] = right_to_left_padding_batch(inputs['input_ids'], padding_token_id=tokenizer.pad_token_id)
    # breakpoint()
    max_length = max_length + inputs['input_ids'].size(1)
    outputs = model.generate(
        **inputs, 
        pad_token_id=tokenizer.eos_token_id,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    # breakpoint()
    if model_type == 'blm':
        # Reverse the output token IDs to get the correct output
        outputs = torch.flip(outputs, dims=[1])
    outputs[outputs>tokenizer.vocab_size] = 0
    generated_texts = [tokenizer.decode(output, skip_special_tokens=False) for output in outputs]
    return generated_texts

def classify_direct(model, tokenizer, prompt, labels, device):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    label_tokens = [tokenizer(label, return_tensors='pt')['input_ids'].to(device) for label in labels]
    outputs = model(**inputs)
    logits = outputs.logits
    choices = torch.stack([torch.sum(logits[:, label[0]], dim=1) for label in label_tokens])
    chosen_index = torch.argmax(choices, dim=0).item()
    return chosen_index


def classify_probabilistic(model, tokenizer, prompt, labels, device, model_type):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    label_tokens = [tokenizer(label, return_tensors='pt')['input_ids'].to(device) for label in labels]
    label_probs = []
    for label_token in label_tokens:
        # Concatenate prompt and label
        if model_type == 'blm':
            # Reverse the input token IDs for the backward language model
            inputs['input_ids'] = torch.flip(inputs['input_ids'], dims=[1])
            label_token = torch.flip(label_token, dims=[1])
        combined_input = torch.cat((inputs['input_ids'], label_token), dim=1)
        combined_input = combined_input.to(device)
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = model(input_ids=combined_input)
        
        # Extract logits for the label part
        logits = outputs.logits[:, -label_token.size(1):, :]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Compute the average probability of the label sequence
        avg_prob = torch.mean(probabilities[0, range(label_token.size(1)), label_token[0]]).item()
        label_probs.append(avg_prob)
    print(label_probs)
    chosen_index = torch.argmax(torch.tensor(label_probs)).item()
    return chosen_index

def create_demo_prompt(examples, obs1, obs2, model_type):
    lbs = [ex[f"hyp{ex['label']}"] for ex in examples]
    if model_type == 'flm':
        demos = '\n'.join([f'Events: Given that "{ex["obs1"]}", and then "{ex["obs2"]}" happened as a result. Reason: {lb}' for ex, lb in zip(examples, lbs)])
        return f'{demos} Given that "{obs1}", and then "{obs2}" happened as a result. Reason: '
    elif model_type == 'blm':
        demos = '\n'.join([f'Reason: {lb} Events: At first, "{ex["obs1"]}", and then due to this reason, "{ex["obs2"]}"' for ex, lb in zip(examples, lbs)])
        return f' Events: At first, "{obs1}", and then due to this reason, "{obs2}"\n{demos}'

        # demos = ' '.join([f'Reason: {lb} Given the reason, these two events happened: "{ex["obs1"]}" at first and then "{ex["obs2"]}".' for ex, lb in zip(examples, lbs)])
        # return f' Given the reason, these two events happened: "{obs1}" at first and then "{obs2}". {demos}'

def postprocess(text, examples, model_type):
    lbs = [ex[f"hyp{ex['label']}"] for ex in examples]
    if model_type == 'flm':
        demos = ' '.join([f'Events: Given that "{ex["obs1"]}", and then "{ex["obs2"]}" happened as a result. Reason: {lb}' for ex, lb in zip(examples, lbs)])
        text = text.replace(demos, '')
    elif model_type == 'blm':
        demos = ' '.join([f'Reason: {lb} Events: At first, "{ex["obs1"]}", and then due to this reason, "{ex["obs2"]}"' for ex, lb in zip(examples, lbs)])
        text = text.replace(demos, '')
    # lbs = [ex[f"hyp{ex['label']}"] for ex in examples]
    # if model_type == 'flm':
    #     demos = ' '.join([f'Given the following sequence of events: "{ex["obs1"]}" followed by "{ex["obs2"]}", the underlying reason for these events is because {lb}' for ex, lb in zip(examples, lbs)])
    #     text = text.replace(demos, '')
    # elif model_type == 'blm':
    #     demos = ' '.join([f'At first, "{ex["obs1"]}" happened and then due to the given reason, "{ex["obs2"]}" happened. Reason: {lb}' for ex, lb in zip(examples, lbs)])
    #     text = text.replace(demos, '')
    return text
        


def make_generation_prompt(obs1, obs2, model_type):
    if model_type == 'flm':
        return f'Given the following sequence of events: \"{obs1}\" followed by \"{obs2}\", the underlying reason for these events is because'
    elif model_type == 'blm':
        return f' Given the reason, these two events happened: \"{obs1}\" at first and then \"{obs2}\".'

def make_classification_prompt(obs1, obs2, model_type):
    if model_type == 'flm':
        prompt = f'Given the following sequence of events: \"{obs1}\" followed by \"{obs2}\", the underlying reason for these events is because '
        return prompt
    elif model_type == 'blm':
        prompt = f' Given the reason, these two events happened: \"{obs1}\" at first and then \"{obs2}\".'
        return prompt

def main(args):
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    # breakpoint()
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
        print(f'the tokenizer pad token id is {tokenizer.pad_token_id}, the token is {tokenizer.pad_token}')
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 2
        print(f'the tokenizer eos token id is {tokenizer.eos_token_id}, the token is {tokenizer.eos_token}')
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)

    results = {}
    if args.num_examples > 1:
        valid_examples = load_valid_examples(args.valid_jsonl_file, args.num_examples)
    if args.task == 'generation':
        generation_data = load_jsonl(args.jsonl_input_file, args.num_samples)
        results['generation'] = []

        batch_size = args.batch_size
        if args.test_method == 'zero_shot':
            prompts = [make_generation_prompt(item['obs1'], item['obs2'], args.model_type) for item in generation_data]
        elif args.test_method == 'demo':
            prompts = [create_demo_prompt(valid_examples, item['obs1'], item['obs2'], args.model_type) for item in generation_data]

        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            batch_generated_texts = generate_text(model, tokenizer, batch_prompts, device, args.model_type)
            for j, prompt in enumerate(batch_prompts):
                results['generation'].append({'prompt': prompt, 'generated_text': postprocess(batch_generated_texts[j], valid_examples, args.model_type)})
    
    elif args.task == 'classification':
        classification_data = load_jsonl(args.jsonl_classification_file, args.num_samples)
        labels = load_txt(args.txt_labels_file)
        results['classification_direct'] = []
        results['classification_probabilistic'] = []
        correct_prob = 0

        for i, item in tqdm(enumerate(classification_data)):
            if args.test_method == 'zero_shot':
                prompt = make_classification_prompt(item['obs1'], item['obs2'], args.model_type)
            elif args.test_method == 'demo':
                prompt = create_demo_prompt(valid_examples, item['obs1'], item['obs2'], args.model_type)
            gold_label = int(labels[i])-1
            prob_choice = classify_probabilistic(model, tokenizer, prompt, [item['hyp1'], item['hyp2']], device, args.model_type)
            if prob_choice == gold_label:
                correct_prob += 1
            results['classification_probabilistic'].append({'prompt': prompt, 'probabilistic_choice': prob_choice, 'gold_label': gold_label, 'hyp1': item['hyp1'], 'hyp2': item['hyp2']})
        
        accuracy_prob = correct_prob / len(classification_data)
        results['accuracy_prob'] = accuracy_prob
        print(f"Probabilistic classification accuracy: {accuracy_prob * 100:.2f}%")
    
    save_json(results, args.json_output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AutoModelForCausalLM model")
    parser.add_argument('--model_name', type=str, required=True, help="Path to the model")
    parser.add_argument('--model_type', type=str, required=True, choices=['blm', 'flm'], help="Type of the model")

    parser.add_argument('--test_method', type=str, required=True, choices=['zero_shot', 'demo'], help="Method to test the model")
    parser.add_argument('--task', type=str, choices=['generation', 'classification'], required=True, help="Task to perform: 'generation' or 'classification'")

    parser.add_argument('--jsonl_input_file', type=str, help="Path to the generation input JSONL file")
    parser.add_argument('--jsonl_classification_file', type=str, help="Path to the classification input JSONL file")
    parser.add_argument('--txt_labels_file', type=str, help="Path to the classification labels TXT file")
    parser.add_argument('--valid_jsonl_file', type=str, required=True, help="Path to the validation input JSONL file for few-shot examples")
    parser.add_argument('--json_output_file', type=str, required=True, help="Path to the output JSON file")

    parser.add_argument('--use_gpu', action='store_true', help="Use GPU for inference if available")
    parser.add_argument('--num_samples', type=int, help="Number of samples to test")
    parser.add_argument('--num_examples', type=int, default=5, help="Number of few-shot examples to use")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for generation task")

    args = parser.parse_args()

    main(args)