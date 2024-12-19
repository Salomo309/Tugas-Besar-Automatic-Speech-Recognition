import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load the fine-tuned model and processor
model_path = "../models/whisper-finetuned/"
model = WhisperForConditionalGeneration.from_pretrained(model_path)
processor = WhisperProcessor.from_pretrained(model_path)

def transcribe_audio(audio_file_path):
    try:
        # Load audio file
        waveform, sampling_rate = torchaudio.load(audio_file_path)
        
        # Resample audio to 16kHz if needed
        if sampling_rate != 16000:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            waveform = resample_transform(waveform)
        
        # Preprocess audio into input features
        input_features = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt").input_features
        
        # Generate transcription
        predicted_ids = model.generate(input_features)
        transcription = processor.decode(predicted_ids[0])
        
        return transcription
    
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
audio_path = "../data/audio/test_audio.wav"
transcription = transcribe_audio(audio_path)
print("Transcribed Text:", transcription)
