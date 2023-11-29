from glob import glob
import torch
import argparse
from tqdm import tqdm
import time
from transformers import SeamlessM4TForSpeechToText, AutoProcessor
from datasets import Audio, Dataset
import os
import json

ARRAY_SIZE = 1 # BATCH ARRAY SIZE
SAMPLING_RATE = 16000

OUTDIR = "../data/transcribed/"
# fp_subset = glob("..data/unlabelled_data/el/2009/*.ogg")
fp_subset = glob("/data/unlabelled_data/el/2016/*.ogg")

class TimeBlock:
    def __init__(self, string=""):
        self.string = string
        
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.string} executed in {self.elapsed_time:.4f} seconds")


def load_model(model_name="facebook/hf-seamless-m4t-large"):
    model = SeamlessM4TForSpeechToText.from_pretrained(model_name)
    model = model.to(torch.device("cuda"))
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def get_data(fps, n):
    return Dataset.from_dict(
        {
            "audio" : fps[:n]
        }
    ).cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE) # hardcoded
    )

def batch_generator(dataset, batch_size):
    for i in range(len(dataset)//batch_size):
        arrays, paths = [], []
        for a in dataset[i*batch_size : (i+1)*batch_size]["audio"]:
            arrays.append(a['array'])
            paths.append(a['path'])
        # list of paths list of arrays
        yield arrays, paths

def decode(model, processor, arrays, tgt_lang='ell'):
    with torch.no_grad():
        inputs = processor(audios=arrays, return_tensors='pt', tgt_lang=tgt_lang, padding=True, truncate=True, requires_grad=False, sampling_rate=SAMPLING_RATE) # hardcoded sampling rate
        inputs = inputs.to(torch.device("cuda"))
        out = model.generate(input_features=inputs['input_features'], tgt_lang=tgt_lang)
        return processor.batch_decode(out, skip_special_tokens=True)

def write_outputs(outputs, filepaths, out_dir, out_name):
    combined = {o:f for o,f in zip(outputs,filepaths)}
    with open(os.path.join(out_dir, out_name), "w") as f:
        json.dump(combined, f)

def transcribe(n=100, bs=5):
    with TimeBlock("Model loading"):
        model, processor = load_model() # default

    with TimeBlock(f"Data Getting for {n}"):
        data = get_data(fp_subset, n)

    with TimeBlock(f"Forward with batchsize: {bs}"):
        for batch in batch_generator(data, bs):
            with TimeBlock("Batch"):
                print(decode(model, processor, batch, 'ell'))

def main(args):
    with TimeBlock("Model Loading"):
        model, processor = load_model()
    with TimeBlock("Data Loading"):
        data = get_data(fp_subset, args.n)
    all_outputs, all_filepaths = [], []
    with TimeBlock("ALL TRANSCRIPTIONS"):
        for array, filepaths in tqdm(batch_generator(data, args.batch_size)):
            outputs = decode(model, processor, array, args.lang)
            all_outputs.extend(outputs)
            all_filepaths.extend(filepaths)
    write_outputs(all_outputs, all_filepaths, OUTDIR, str(args.index)+"out.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, help="Index for batching 1-N")
    parser.add_argument("--n", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lang", type=str)
    args = parser.parse_args()
    main(args)
