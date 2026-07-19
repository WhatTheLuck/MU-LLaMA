import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
from scipy.io import wavfile
import os
import hashlib
from pathlib import Path
import numpy as np
from data.utils import *
import torchaudio

from features.dissonance_adapter import DissonanceFeatureAdapter

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

class FinetuneDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None,
                 audio_root="../MusicQA/audios", dissonance_config=None, return_metadata=False):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        ann = []
        for meta_path in self.config['META']:
            resolved_meta = Path(meta_path)
            if not resolved_meta.is_absolute() and not resolved_meta.is_file():
                resolved_meta = (Path(config_path).resolve().parent / resolved_meta).resolve()
            meta_l = json.load(open(resolved_meta, encoding="utf-8"))
            print(f"{meta_path}: len {len(meta_l)}")
            ann += meta_l
        self.ann = ann
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.audio_root = Path(audio_root)
        self.return_metadata = bool(return_metadata)
        ds_config = dict(dissonance_config or {})
        self.dissonance = DissonanceFeatureAdapter(ds_config) if ds_config.get("enabled", False) else None
        self.clip_sampler = ConstantClipsPerVideoSampler(
                clip_duration=2, clips_per_video=3)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item = self.ann[index]
        if "audio_name" in data_item or "audio_path" in data_item:
            filename = Path(data_item.get("audio_path") or data_item['audio_name'])
            if not filename.is_absolute():
                filename = self.audio_root / filename
            filename = filename.resolve()
            question = data_item['conversation'][0]['value']
            answer = data_item['conversation'][1]['value']
            sample_rate = 24000
            waveform, sr = torchaudio.load(str(filename))
            if sample_rate != sr:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
            image = torch.mean(waveform, 0)
            has_audio = True
        else:
            question = data_item['conversation'][0]['value']
            answer = data_item['conversation'][1]['value']
            image = torch.zeros(24000)
            filename = None
            sample_rate = 24000
            has_audio = False
        format_instruction = question
        input1 = llama.utils.format_prompt(format_instruction)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        if not self.return_metadata and self.dissonance is None:
            return input2, labels, input2_mask, image
        audio_id = str(data_item.get("audio_id") or data_item.get("audio_name") or f"record_{index}")
        question_id = str(data_item.get("question_id") or hashlib.sha1(
            f"{audio_id}\0{question}".encode("utf-8")
        ).hexdigest()[:16])
        ds_payload = None
        if self.dissonance is not None and has_audio:
            ds_payload = self.dissonance.load(audio_id, filename)
        elif self.dissonance is not None:
            ds_payload = {
                "dissonance": torch.zeros(self.dissonance.frequency_bins, 1),
                "energy": None,
                "metadata": {"valid": False},
            }
        metadata = {
            "audio_id": audio_id,
            "question_id": question_id,
            "question": question,
            "question_type": data_item.get("question_type") or data_item.get("type") or "unknown",
            "reference": answer,
            "audio_path": str(filename) if filename is not None else None,
            "sample_rate": sample_rate,
            "has_audio": has_audio,
            "dissonance_payload": ds_payload,
        }
        return input2, labels, input2_mask, image, metadata


def finetune_collate(batch):
    examples, labels, masks, waveforms, metadata = zip(*batch)
    max_audio = max(waveform.numel() for waveform in waveforms)
    audio_batch = waveforms[0].new_zeros((len(waveforms), max_audio))
    audio_lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
    for index, waveform in enumerate(waveforms):
        audio_batch[index, :waveform.numel()] = waveform

    ds_payloads = [item.get("dissonance_payload") for item in metadata]
    ds_batch = None
    ds_mask = None
    if any(payload is not None for payload in ds_payloads):
        first = next(payload for payload in ds_payloads if payload is not None)
        frequency_bins = int(first["dissonance"].shape[0])
        max_frames = max(
            int(payload["dissonance"].shape[1]) if payload is not None else 1
            for payload in ds_payloads
        )
        ds_batch = torch.zeros(len(batch), frequency_bins, max_frames, dtype=torch.float32)
        ds_mask = torch.zeros(len(batch), max_frames, dtype=torch.bool)
        for index, payload in enumerate(ds_payloads):
            if payload is None or payload.get("metadata", {}).get("valid") is False:
                continue
            spectrum = payload["dissonance"].float()
            if spectrum.shape[0] != frequency_bins:
                raise ValueError("All DS features in a batch must use the same frequency bins")
            ds_batch[index, :, :spectrum.shape[1]] = spectrum
            ds_mask[index, :spectrum.shape[1]] = True

    extras = {
        "audio_lengths": audio_lengths,
        "dissonance": ds_batch,
        "dissonance_mask": ds_mask,
        "metadata": [{key: value for key, value in item.items() if key != "dissonance_payload"}
                     for item in metadata],
    }
    return torch.stack(examples), torch.stack(labels), torch.stack(masks), audio_batch, extras
