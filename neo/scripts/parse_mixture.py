
"""
This script is used to parse the mixture of the pretraining data
input: path to the yaml file
output: a megatron style data mixture string
"""

import os
import argparse
import yaml

# /data/public_models/huggingface/matrix/tmp/rr_code_code.000_text_document: 43902478081
# /data/public_models/huggingface/matrix/tmp/rr_exam_math_text_document: 51337646588
# /data/public_models/huggingface/matrix/tmp/rr_paper_math.000_text_document: 37748191563
# /data/public_models/huggingface/matrix/tmp/rr_code_code.002_text_document: 43824673654
# /data/public_models/huggingface/matrix/tmp/rr_cc_math.000_text_document: 14273576155
# /data/xunjian_yin/mycode/DCLM/dclm-b1/dclm_reverse/dclm_02_text_document: 243270290008

DB2TOKCNT = {
    '/data/xunjian_yin/mycode/DCLM/dclm-b1/dclm_reverse/dclm_01_text_document.bin': 288703527565,
    '/data/public_models/huggingface/matrix/tmp/rr_code_code.000_text_document.bin': 43902478081,
    '/data/public_models/huggingface/matrix/tmp/rr_exam_math_text_document.bin': 51337646588,
    '/data/public_models/huggingface/matrix/tmp/rr_paper_math.000_text_document.bin': 37748191563,
    '/data/public_models/huggingface/matrix/tmp/rr_code_code.002_text_document.bin': 43824673654,
    '/data/public_models/huggingface/matrix/tmp/rr_cc_math.000_text_document.bin': 14273576155,
    '/data/xunjian_yin/mycode/DCLM/dclm-b1/dclm_reverse/dclm_02_text_document.bin': 243270290008,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="parse the mixture of the pretraining data")
    parser.add_argument(
        "--cfg",
        "-c",
        type=str,
        required=True,
        help="path to the yaml file")
    return parser.parse_args()


def load_yaml(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def parse_mixture_from_cfg(cfg):
    keys = list(cfg.keys())
    # find keys ends with _ROUND
    rounds = [k for k in keys if k.endswith("_ROUND")]

    def repeat_str(s, n):
        return "".join([s for _ in range(n)])
    print(rounds)
    total_tokcnt = 0
    mixture_str = ""
    for r in rounds:
        repeat_times = float(r.replace("_ROUND", ""))
        print(cfg[r])
        print(set(cfg[r]))
        mmap_paths = sorted(set(cfg[r]))
        for mmap_path in mmap_paths:
            mmap_path_without_ext = os.path.splitext(mmap_path)[0]
            tokcnt = DB2TOKCNT[mmap_path]
            if isinstance(tokcnt, str):
                assert tokcnt.endswith("B"), f"invalid tokcnt: {tokcnt}"
                tokcnt = float(tokcnt.replace("B", "")) * 10**9
                total_tokcnt += tokcnt * repeat_times
            else:
                assert isinstance(tokcnt, int), f"invalid tokcnt: {tokcnt}"
                total_tokcnt += tokcnt * repeat_times

            mixture_str += f"{int(tokcnt * repeat_times)} {mmap_path_without_ext} "

    # total iter count
    total_iter = total_tokcnt / (cfg["GLOBAL_BATCH_SIZE"] * cfg["SEQ_LEN"])

    # into string x.xxxB
    total_tokcnt /= 1e9
    total_tokcnt = f"{total_tokcnt:.3f}B"

    return mixture_str, total_tokcnt, total_iter


if __name__ == "__main__":
    args = parse_args()

    cfg = load_yaml(args.cfg)
    print(f"[INFO] Loaded cfg from {args.cfg}")

    mixture_str, total_tokcnt, total_iter = parse_mixture_from_cfg(cfg)
    print(f"[INFO] Mixture string: {mixture_str}")
    print(f"[INFO] Total token count: {total_tokcnt}")
    print(f"[INFO] Remember to change TRAIN_ITERS to: {total_iter}")
