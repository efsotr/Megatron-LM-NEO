from transformers import AutoTokenizer
from load_reverse_tokenizer import load_reverse_tokenizer

def test_tokenizer(tokenizer, text, batch=False, padding="longest", truncation=True, max_length=20):
    tokenizer.padding_side = 'left'
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if batch:
        print("Testing batch operations...")
        # Testing batch_encode_plus and batch_decode
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer([text, text[::-1]], padding=padding, truncation=truncation, max_length=max_length, return_tensors='pt', add_special_tokens=True)
        print("Batch Token IDs:", [ids.tolist() for ids in encoded['input_ids']])
        print("Batch Attention Masks:", [mask.tolist() for mask in encoded['attention_mask']])
        decoded = tokenizer.batch_decode(encoded['input_ids'], skip_special_tokens=True)
        print("Batch Decoded Texts:", decoded)
    else:
        print("Testing single operations...")
        # Testing encode_plus and decode
        encoded = tokenizer(text, padding=padding, truncation=truncation, max_length=max_length, return_tensors='pt', add_special_tokens=True)
        print("Token IDs:", encoded['input_ids'][0].tolist())
        print("Attention Mask:", encoded['attention_mask'][0].tolist())
        decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
        print("Decoded Text:", decoded)
    print("\n")

def compare_tokenizers(original_path, texts):
    print("Loading original tokenizer...")
    original_tokenizer = AutoTokenizer.from_pretrained(original_path)
    
    print("Loading custom tokenizer...")
    custom_tokenizer = load_reverse_tokenizer(original_path)

    for text in texts:
        print("Original Tokenizer:")
        test_tokenizer(original_tokenizer, text)
        test_tokenizer(original_tokenizer, text, batch=True)

        print("Custom (Reversed) Tokenizer:")
        test_tokenizer(custom_tokenizer, text)
        test_tokenizer(custom_tokenizer, text, batch=True)

# Define a set of test strings
texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "😊🚀🐍📊",  # emojis
    ""  # empty string
]

# Path to your saved tokenizer
tokenizer_path = '/data/xunjian_yin/mycode/MAP-NEO/Megatron-LM-NEO/hf_checkpoints/7B/142.61B'

# Run the comparison tests
compare_tokenizers(tokenizer_path, texts)