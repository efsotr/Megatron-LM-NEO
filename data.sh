


python tools/preprocess_data.py \
       --input "/data/xunjian_yin/mycode/split_files_03/part_*" \
       --output-prefix /data/xunjian_yin/mycode/DCLM/dclm-b1/dclm_reverse/dclm_03 \
       --tokenizer-model neo/tokenizer.model \
       --tokenizer-type SentencePieceTokenizer \
       --keep-sequential-samples \
	--workers 64 \
       --partitions 32 \
       --append-eod

# python tools/preprocess_data.py \
#        --input "/data/datasets/yxj/code_code.001*" \
#        --output-prefix /data/datasets/yxj/rr_code_code.001 \
#        --tokenizer-model neo/tokenizer.model \
#        --tokenizer-type SentencePieceTokenizer \
#        --keep-sequential-samples \
# 	--workers 100 \
#        --partitions 10 \
#        --append-eod




# python tools/preprocess_data.py \
#        --input "/data/public_models/huggingface/yxj/paper_math.000*" \
#        --output-prefix /data/public_models/huggingface/matrix/rr_paper_math.000 \
#        --tokenizer-model neo/tokenizer.model \
#        --tokenizer-type SentencePieceTokenizer \
#        --keep-sequential-samples \
# 	--workers 99 \
#        --partitions 3 \
#        --append-eod


# python tools/preprocess_data.py \
#        --input "/data/public_models/huggingface/yxj/wiki_*" \
#        --output-prefix /data/public_models/huggingface/matrix/rr_wiki \
#        --tokenizer-model neo/tokenizer.model \
#        --tokenizer-type SentencePieceTokenizer \
#        --keep-sequential-samples \
# 	--workers 100 \
#        --partitions 2 \
#        --append-eod

# python tools/preprocess_data.py \
#        --input "/data/public_models/huggingface/yxj/book_reviews.0000.jsonl" \
#        --output-prefix /data/datasets/yxj/ \
#        --tokenizer-model neo/tokenizer.model \
#        --tokenizer-type SentencePieceTokenizer \
#        --keep-sequential-samples \
# 	--workers 8 \
#        --partitions 1 \
#        --append-eod


# python tools/count_mmap_token.py --mmap_path /data/datasets/yxj/_text_document
# python tools/count_mmap_token.py --mmap_path /data/public_models/huggingface/matrix/tmp/book_all_text_document