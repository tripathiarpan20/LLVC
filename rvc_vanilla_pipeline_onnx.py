import torch
from torch import nn
from minimal_rvc.pipeline import VocalConvertPipeline
from minimal_rvc.models import SynthesizerTrnMs256NSFSidNono
from minimal_quickvc.models import SynthesizerTrn
from minimal_quickvc.utils import load_checkpoint
from fairseq import checkpoint_utils
import librosa
import argparse
import numpy as np
from scipy.io.wavfile import write

def init_model(model_type):
    if model_type == 'rvc':
        model_path = "llvc_models/models/rvc_no_f0/f_8312_no_f0-300.pth"
        state_dict = torch.load(model_path, map_location="cpu")
        state_dict["params"]["spk_embed_dim"] = state_dict["weight"][
            "emb_g.weight"
        ].shape[0]
        if not "emb_channels" in state_dict["params"]:
            state_dict["params"]["emb_channels"] = 768  # for backward compat.
        model = SynthesizerTrnMs256NSFSidNono(
            **state_dict["params"], is_half=False
        ).eval().to('cpu')
        model.load_state_dict(state_dict["weight"], strict=False)
    elif model_type == 'quickvc':
        model = SynthesizerTrn().eval().to('cpu')
        model_path = 'llvc_models/models/quickvc/quickvc_100.pth'
        _ = load_checkpoint(model_path, model, None)
    return model


def load_hubert(model_type):
    if model_type == 'rvc':
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ['llvc_models/models/embeddings/checkpoint_best_legacy_500.pt'],
            suffix="",
        )
        embedder_model = models[0]
        embedder_model = embedder_model.to('cpu')
    if model_type == 'quickvc':
        embedder_model = torch.hub.load(
            "bshall/hubert:main", "hubert_soft").eval().to('cpu')
    return embedder_model


class RVCPipeline(nn.Module):
    def __init__(self):
        super(RVCPipeline, self).__init__()
        #sr_out = 32000 if args.model_type == 'rvc' else 16000
        self.pipeline = VocalConvertPipeline(32000, 'cpu', False, no_pad=True)
        self.model = init_model("rvc")
        self.hubert = load_hubert("rvc")

        self.crossfade_overlap = 256
        self.crossfade_offset_rate = 0.0
        self.crossfade_end_rate = 1.0
        self.generate_strength()

    def generate_strength(self):
        cf_offset = int(self.crossfade_overlap * self.crossfade_offset_rate)
        cf_end = int(self.crossfade_overlap * self.crossfade_end_rate)
        cf_range = cf_end - cf_offset
        percent = np.arange(cf_range) / cf_range

        np_prev_strength = np.cos(percent * 0.5 * np.pi) ** 2
        np_cur_strength = np.cos((1 - percent) * 0.5 * np.pi) ** 2

        self.np_prev_strength = np.concatenate(
            [
                np.ones(cf_offset),
                np_prev_strength,
                np.zeros(self.crossfade_overlap -
                         cf_offset - len(np_prev_strength)),
            ]
        )
        self.np_cur_strength = np.concatenate(
            [
                np.zeros(cf_offset),
                np_cur_strength,
                np.ones(self.crossfade_overlap -
                        cf_offset - len(np_cur_strength)),
            ]
        )

    def clear_buffers(self):
        self.audio_buffer = None
        del self.sola_buffer

    def postprocess(self, audio):
        if hasattr(self, "sola_buffer") is True:
            np.set_printoptions(threshold=10000)
            audio_offset = -1 * (
                self.sola_search_frame + self.crossfade_overlap + self.block_len
            )
            audio = audio[audio_offset:]
            # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC, https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI
            cor_nom = np.convolve(
                audio[: self.crossfade_overlap + self.sola_search_frame],
                np.flip(self.sola_buffer),
                "valid",
            )
            cor_den = np.sqrt(
                np.convolve(
                    audio[: self.crossfade_overlap +
                          self.sola_search_frame] ** 2,
                    np.ones(self.crossfade_overlap),
                    "valid",
                )
                + 1e-3
            )
            sola_offset = int(np.argmax(cor_nom / cor_den))
            sola_end = sola_offset + self.block_len
            output_wav = audio[sola_offset:sola_end].astype(np.float64)
            output_wav[: self.crossfade_overlap] *= self.np_cur_strength
            output_wav[: self.crossfade_overlap] += self.sola_buffer[:]
            result = output_wav.astype(np.int16)

        else:
            # print("[Voice Changer] no sola buffer. (You can ignore this.)")
            result = np.zeros(4096).astype(np.int16)

        if (
            hasattr(self, "sola_buffer") is True
            and sola_offset < self.sola_search_frame
        ):
            offset = -1 * (
                self.sola_search_frame + self.crossfade_overlap - sola_offset
            )
            end = -1 * (self.sola_search_frame - sola_offset)
            sola_buf_org = audio[offset:end]
            self.sola_buffer = sola_buf_org * self.np_prev_strength
        else:
            self.sola_buffer = audio[-self.crossfade_overlap:] * \
                self.np_prev_strength
        return result

    def forward(self, x):
        return self.pipeline(
                    self.hubert,
                    12,
                    self.model,
                    0,
                    x,
                    12,
                    None,
                    None,
                    0,
                    0,
                    True
                )
    def __call__ (self, x):
        return self.forward(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", "-f", type=str,
                        default="Recording.wav", help="audio file to convert")
    parser.add_argument("--out_onnx", "-o", type=str,
                        default="onnx_model.onnx", help="output onnx file")
    args = parser.parse_args()
    #self.conv_sr = 16000 in `compare_infer.py`
    conv_sr = 16000
    sr_out = 32000
    model = RVCPipeline()

    input_audio, sr = librosa.load(args.fname, sr=conv_sr)
    print('LOG: input audio shape: ', input_audio.shape)
    output_audio = model(torch.Tensor(input_audio))
    print('LOG: output audio shape: ', output_audio.shape)
    postprocessed = model.postprocess(output_audio)
    print('LOG: postprocessed output audio shape: ', output_audio.shape)
    write('output.wav', sr_out, postprocessed)

    torch.onnx.export(model,               # model being run
                  torch.Tensor(input_audio),                         # model input (or a tuple for multiple inputs)
                  args.out_onnx,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  # opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {1 : 'batch_size'},    # variable length axes
                                'output' : {1 : 'batch_size'}}
                  )

    print(model)