import argparse
# import logging
import torch
from glob import glob
# import torchaudio
# expects seamless_communication dir to exist
from seamless_communication.models.inference import Translator

DEVICE = torch.cuda()



def main(args):
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Audio WAV file path or text input.")
    parser.add_argument('--model', 
                      type=str,
                      help="Base model name (`seamlessM4T_medium`, `seamlessM4T_large`)", 
                      default="seamlessM4T_large"
                    )
    parser.add_argument(
        "--vocoder_name", type=str, help="Vocoder name", default="vocoder_36langs"
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        help="Source language, only required if input is text.",
        default=None,
    )
    parser.add_argument("task", type=str, help="Task type, defaults to speech transcription s2tt", default="s2tt")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
        # logger.info(f"Running inference on the GPU in {dtype}.")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        # logger.info(f"Running inference on the CPU in {dtype}.")
    
    args = parser.parse_args()
    args.device = device
    args.dtype = dtype
