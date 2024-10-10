import numpy as np
import struct
import os
from megatron.core.datasets import indexed_dataset


# Assuming the classes DType, _IndexReader, and _IndexWriter are defined as in the indexed_dataset.py script.

def reverse_indices_and_lengths(idx_reader, output_idx_path, output_bin_path):
    # Reverse the document ids and corresponding sentence lengths
    reversed_doc_ids = []
    reversed_sentence_lens = []

    # Read the original data
    for i in range(len(idx_reader.sequence_lengths)):
        start_ptr = idx_reader.sequence_pointers[i]
        end_ptr = start_ptr + idx_reader.sequence_lengths[i] * idx_reader.dtype_size
        doc_ids = np.frombuffer(idx_reader.bin_buffer[start_ptr:end_ptr], dtype=np.int16)
        breakpoint()
        # Reverse the doc_ids and corresponding sentence_lens
        reversed_doc_ids.append(doc_ids[::-1])
        reversed_sentence_lens.append(idx_reader.sequence_lengths[i])

    # Write the reversed data to new .bin and .idx files
    with open(output_bin_path, 'wb') as bin_writer, indexed_dataset._IndexWriter(output_idx_path, idx_reader.dtype) as idx_writer:
        sequence_pointers = []
        curr_ptr = 0
        
        for doc_ids in reversed_doc_ids:
            bin_writer.write(doc_ids.tobytes(order="C"))
            sequence_pointers.append(curr_ptr)
            curr_ptr += len(doc_ids) * idx_reader.dtype_size
        
        # Write the new index file
        idx_writer.write(reversed_sentence_lens, None, sequence_pointers)

# Define file paths
input_idx_path = '/data/public_models/huggingface/matrix/tmp/r_instruction_all_text_document.idx'
input_bin_path = '/data/public_models/huggingface/matrix/tmp/r_instruction_all_text_document.bin'
output_idx_path = '/data/public_models/huggingface/matrix/tmp/rr_instruction_all_text_document.idx'
output_bin_path = '/data/public_models/huggingface/matrix/tmp/rr_instruction_all_text_document.bin'

# Initialize the index reader
idx_reader = indexed_dataset._IndexReader(input_idx_path, multimodal=False)

# Reverse indices and lengths and write to new files
reverse_indices_and_lengths(idx_reader, output_idx_path, output_bin_path)