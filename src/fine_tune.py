import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils import preprocess_dataset, create_dataset

model_name = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

dataset = create_dataset("..\\data\\audio", "data\\transcript.txt")
dataset = Dataset.from_list(dataset)

dataset = preprocess_dataset(dataset, processor)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="../models/checkpoints/",
    evaluation_strategy="epoch",
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
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
)

# Train model
trainer.train()
model.save_pretrained("../models/whisper-finetuned/")
