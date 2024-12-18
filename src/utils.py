import torchaudio
import os
from transformers import WhisperProcessor

def create_dataset (audio_path, transcript_path):
    audio_files = sorted([os.path.join(audio_path, f) for f in os.path.abspath(audio_path) if f.endswith('.wav')])
    with open(os.path.abspath(transcript_path), "r") as f:
        transcriptions = [line.strip() for line in f.readlines()]

    dataset = []
    for audio_file, transcription in zip(audio_files, transcriptions):
        dataset.append({"audio": audio_file, "sentence": transcription})
    return dataset


def preprocess_dataset(dataset, processor):
    def preprocess(batch):
        audio = batch["audio"]
        input_features = processor(audio["array"], sampling_rate=16000, return_tensors="pt").input_features
        batch["input_features"] = input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    return dataset.map(preprocess, remove_columns=["audio", "sentence"])

def compute_metrics(predictions, references):
    from jiwer import wer
    return {"wer": wer(references, predictions)}

