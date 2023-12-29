from model import VoiceConvertModel
import sys
import logging
from pydub import AudioSegment
from tqdm import tqdm
from typing import Optional
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import os
import argparse

logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)



def resample_file_(wav_path, target_sr, out_path):
    y, sr = librosa.load(wav_path, sr=target_sr)
    sf.write(out_path, y, sr)
    return True


def convert_file(in_path: Path | str,
                   model_path: Path | str,
                   model_name: str,
                   f0_up_key: int,
                   f0_method: str = 'rmvpe',
                   converted_path: Optional[Path | str] = None) -> None:
    model = VoiceConvertModel(
        model_name, torch.load(model_path, map_location="cpu"))
    # convert and save (explicitly typed to avoid misconstruing the audio type for something else)
    out = model.single(sid=1,
                                    input_audio=str(in_path),
                                    embedder_model_name='hubert_base',
                                    embedding_output_layer='auto',
                                    f0_up_key=f0_up_key,
                                    f0_file=None,
                                    f0_method=f0_method,
                                    auto_load_index=False,
                                    faiss_index_file=None,
                                    index_rate=None,
                                    f0_relative=True,
                                    output_dir='out')

    out.export(converted_path, format="wav")

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--root_path", help="Path to dataset to be processed", type=str, required=True, default='.LibriSpeech/')
    parser.add_argument("--input_file", help="Path to the input wav file",
                        type=str, required=True)
    parser.add_argument("--out_dir", help="Dir to save the output",
                        type=str, required=True, default='./LibriSpeech_processed/')
    parser.add_argument(
        "--model_path", help="Path to RVC .pth model file", type=str)
    parser.add_argument(
        "--f0_up_key", help="Pitch adjust for conversion", type=int, default=12)
    parser.add_argument(
        "--f0_method", help="f0 method for pitch extraction, only `rmvpe` supported tho", type=str, default='rmvpe')
    parser.add_argument("--model_name", help="Name to be added to output filenames. If `None`, the filename of --model_path will be used",
                        type=str, required=False, default=None)
    parser.add_argument(
        "--target_sr", help="Sample rate to resample the dataset into", type=int, default=16000)
    parser.add_argument(
        "--random_seed", help="Random seed for splitting the dataset", type=int, default=42)
    args = parser.parse_args()

    # unpack, typecheck, and perform safety checks on args
    out_dir = Path(args.out_dir)
    model_path = Path(args.model_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    assert model_path.exists(), f'{model_path} does not exist'

    model_name: str = args.model_name
    if model_name is None:
        model_name = model_path.name.split('.')[0]

    target_sr: int = args.target_sr
    f0_up_key: int = args.f0_up_key

    assert isinstance(target_sr, int), f'target_sr must be an integer'
    assert target_sr > 0, f'target_sr must be greater than 0'

    # resample the converted files to the target_sr
    print("[INFER_FILE] Resampling converted files")
    resample_file_(wav_path= args.input_file,
                target_sr=target_sr,
                out_path=os.path.join(out_dir, 'in_resampled.wav'),
                )

    print(f'[INFER_FILE] Converting {args.input_file}')
    convert_file(os.path.join(out_dir, 'in_resampled.wav'),
                    model_path=model_path,
                    model_name=model_name,
                    f0_up_key=f0_up_key,
                    f0_method=args.f0_method,
                    converted_path=os.path.join(out_dir, 'out.wav'))
    

if __name__ == '__main__':
    main()
