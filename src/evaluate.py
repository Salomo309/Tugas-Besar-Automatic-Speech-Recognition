from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset
from utils import compute_metrics, create_dataset

model = WhisperForConditionalGeneration.from_pretrained("../models/whisper-finetuned/")
processor = WhisperProcessor.from_pretrained("../models/whisper-finetuned/")

test_data = create_dataset("../data/test", "../data/transcripts_test.txt")
test_data = Dataset.from_list(test_data)

predictions, references = [], []
for example in test_data:
    input_features = processor(example["audio"], sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.decode(predicted_ids[0])
    predictions.append(transcription)
    references.append(example["sentence"])

metrics = compute_metrics(predictions, references)
print("Evaluation Metrics:", metrics)
