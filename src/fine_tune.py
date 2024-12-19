import os
import torch
from sklearn.model_selection import train_test_split
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, DatasetDict
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils import preprocess_dataset, create_dataset

class DataCollatorWhisper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_features = torch.stack(
            [torch.tensor(example["input_features"]) if not isinstance(example["input_features"], torch.Tensor) else example["input_features"] for example in batch]
        )
        
        labels = [
            torch.tensor(example["labels"], dtype=torch.long) if not isinstance(example["labels"], torch.Tensor) else example["labels"]
            for example in batch
        ]
        
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        
        return {
            "input_features": input_features,
            "labels": labels,
        }

model_name = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

dataset = create_dataset("..\\Tugas-Besar-Automatic-Speech-Recognition\\data\\audio", "data\\transcript.txt")
dataset = Dataset.from_list(dataset)
dataset = preprocess_dataset(dataset, processor)

data_collator = DataCollatorWhisper(processor)

train_dataset = dataset
eval_dataset = dataset

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="../models/checkpoints/",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    save_strategy="epoch",
    predict_with_generate=True,
    fp16=True,
)

# Define trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=None,
    data_collator=data_collator,
)

# Train model
trainer.train()
model.save_pretrained("..\\Tugas-Besar-Automatic-Speech-Recognition\\models\\whisper-finetuned")
