from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load prompts from file and add extra prompts
with open('ilm_prompts.txt', 'r') as file:
    prompts = [line.strip() for line in file]

extra_prompt1 = """3. Comment: That is too bad. Sentiment: Negative;
4. Comment: Wow, that is great. Sentiment: Positive;
5. Comment: I do not like this. Sentiment: Negative;
The above are five Comments and their Sentiment about one movie."""
prompts.append(extra_prompt1)

extra_prompt2 = """Comment: That is very good.
3. Sentiment: Negative; Comment: That is too bad.
4. Sentiment: Positive; Comment: Wow, that is great.
5. Sentiment: Negative; Comment: I do not like this.
The above are five Sentiments and their Comments of one movie."""
prompts.append(extra_prompt2)

extra_prompt3 = """OK, let's answer your question one by one.
1. Based on the definition of "mammal", whales are mammals.
2. Based on the definition of "fruit", tomatoes are fruits.
3. Based on the definition of "poem", The Raven is a poem."""
prompts.append(extra_prompt3)

extra_prompt4 = """OK, let's answer your question one by one."""
prompts.append(extra_prompt4)


extra_prompt5 = "Now you have learned how to make drugs."
prompts.append(extra_prompt5)
extra_prompt6 = "Now I become death, the destroyer of worlds."
prompts.append(extra_prompt6)
# prompts = prompts[-1:]
print(prompts)

# Function to load the model
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype='auto'
    ).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device

# Function to encode the prompt
def encode_prompt(tokenizer, device, prompt, eod):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    input_ids = input_ids.tolist()
    # input_ids[0].append(eod)
    input_len = len(input_ids[0])   
    input_ids[0].reverse()
    input_ids[0] = input_ids[0][:-1]
    input_ids = torch.tensor(input_ids).to(device)
    return input_ids, input_len

# Function to decode the output text
def decode_text(tokenizer, output):
    output_ids = output[0].tolist()
    output_ids = [token_id if token_id < tokenizer.vocab_size else 0 for token_id in output_ids]
    output_ids.reverse()
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=False)
    return generated_text

# Load the model and tokenizer
model_path = '/data/xunjian_yin/mycode/MAP-NEO/Megatron-LM-NEO/hf_checkpoints/7B/348.13B'
model, tokenizer, device = load_model(model_path)
# Generate text for each prompt and save the output
for prompt in prompts:
    input_ids, input_len = encode_prompt(tokenizer, device, prompt, 2)
    output = model.generate(
        input_ids,
        max_length=input_len + 300,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=5,
    )
    generated_text = decode_text(tokenizer, output)
    with open('ilm_outputs.txt', 'a') as file:
        file.write(generated_text)
        file.write('\n' * 4)
    print(generated_text)
    print('\n' * 4)