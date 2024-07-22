from functools import partial
import os
import fire
import tqdm
import json
from transformers import AutoTokenizer
import multiprocessing
import glob
import random
import pandas as pd

END_OF_SYSTEM_PROMPT = "<|endofsystemprompt|>"
BEGIN_OF_SYSTEM = "<|beginofsystem|>"
BEGIN_OF_USER = "<|beginofuser|>"
END_OF_CHAT = "<|endofchat|>"
CONTEXT_LENGTH = 8192
MP_CHUNKSIZE = 1024


def tokenize_text(text, tokenizer):
    text = [text]
    for special_token in tokenizer.all_special_tokens:
        text_upd = []
        for cut in text:
            for i, t in enumerate(cut.split(special_token)):
                if i != 0:
                    text_upd.append(special_token)
                text_upd.append(t)
        text = text_upd

    token_ids = []
    for cut in text:
        token_ids.extend(tokenizer(cut, add_special_tokens=False)["input_ids"])

    return token_ids


def k2_tokenize_example(datapoint, tokenizer):
    token_ids, tgt_mask = [], []

    # input masks
    for uttr in datapoint["seed"]:
        if uttr["role"] == "system":
            # uttr_text = str(uttr['content']) + END_OF_SYSTEM_PROMPT
            continue
        elif uttr["role"] == "user":
            uttr_text = BEGIN_OF_USER + str(uttr["content"])
        elif uttr["role"] == "assistant":
            uttr_text = BEGIN_OF_SYSTEM + str(uttr["content"]) + tokenizer.eos_token
        else:
            raise ValueError
        uttr_token_ids = tokenize_text(text=uttr_text, tokenizer=tokenizer)
        token_ids.extend(uttr_token_ids)
        tgt_mask.extend([0] * len(uttr_token_ids))

    # output
    uttr_text = BEGIN_OF_SYSTEM + str(datapoint["response"]) + tokenizer.eos_token
    uttr_token_ids = tokenize_text(text=uttr_text, tokenizer=tokenizer)

    token_ids.extend(uttr_token_ids)
    tgt_mask.append(0)
    tgt_mask.extend([1] * (len(uttr_token_ids) - 1))

    token_ids.append(tokenizer.vocab[END_OF_CHAT])
    tgt_mask.append(0)
    assert len(token_ids) == len(tgt_mask)

    return {"token_ids": token_ids, "tgt_mask": tgt_mask}



def main():

    all_training_data_with_response = pd.read_json(
        "all_training_data_with_response.jsonl", lines=True
        )

    tokenizer = AutoTokenizer.from_pretrained(
        "D:/Project/k2-safety/process/k2_tokenizer/tokenizer_function_call"
    )
    tokenize_fn = partial(k2_tokenize_example, tokenizer=tokenizer)

    for special_token in tokenizer.all_special_tokens:
        print(
            f"{special_token}:",
            tokenizer(special_token, add_special_tokens=False)["input_ids"],
        )

    datapoints = all_training_data_with_response.to_dict(orient="records")

    tokenized_rows = []
    with multiprocessing.Pool(16) as pool:
        new_examples = list(
            tqdm.tqdm(
                pool.imap_unordered(tokenize_fn, datapoints, chunksize=MP_CHUNKSIZE),
                total=len(datapoints),
            )
        )
        tokenized_rows.extend(new_examples)

    pd.DataFrame(tokenized_rows).to_json(
        "all_training_data_with_response_tokenized.jsonl", lines=True, orient="records"
    )


if __name__ == "__main__":
    main()
