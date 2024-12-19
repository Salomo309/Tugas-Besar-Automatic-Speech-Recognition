import torchaudio
import os
from transformers import WhisperProcessor
import torch
from datasets import Dataset

def create_dataset(audio_path, transcript_path):
    audio_files = sorted([os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith('.wav') or f.endswith('.mp3')])
    with open(os.path.abspath(transcript_path), "r") as f:
        transcriptions = [line.strip() for line in f.readlines()]
    print(f"Audio Files: {audio_files}")

    dataset = []
    for audio_file, transcription in zip(audio_files, transcriptions):
        # Load audio data
        waveform, sample_rate = torchaudio.load(audio_file)
        dataset.append({"audio": {"array": waveform.squeeze(0).numpy(), "sampling_rate": sample_rate}, "sentence": transcription})
    
    return dataset


def preprocess_dataset(dataset, processor):
    def preprocess(batch):
        audio = batch["audio"]
        input_features = processor(audio["array"], sampling_rate=16000, return_tensors="pt").input_features[0]
        input_ids = processor.tokenizer(batch["sentence"]).input_ids
        
        return {
            "input_features": input_features,
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }

    return dataset.map(preprocess, remove_columns=["audio", "sentence"])


def compute_metrics(predictions, references):
    from jiwer import wer
    return {"wer": wer(references, predictions)}
