from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel
import os
import torch
import numpy as np
import pandas as pd
import gensim
import io
import speech_recognition as sr
import noisereduce as nr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from fuzzywuzzy import process
import torch.nn as nn
import torch.optim as optim

app = FastAPI()

# Load CSV File
df = pd.read_csv("Classtextfinal.csv")
commands = df["Transcription"].astype(str).tolist()
labels = df["Class"].astype(str).tolist()
labels.append("unknown_command")

# Tokenization
commands_tokenized = [cmd.split() for cmd in commands]

# Train Word2Vec Model
embedding_dim = 50
w2v_model = Word2Vec(sentences=commands_tokenized, vector_size=embedding_dim, window=5, min_count=1, workers=4)

def text_to_vector(text):
    words = text.split()
    vec = np.zeros(embedding_dim)
    count = 0
    for word in words:
        if word in w2v_model.wv:
            vec += w2v_model.wv[word]
            count += 1
    return vec / count if count > 0 else vec

X = np.array([text_to_vector(cmd) for cmd in commands])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(1))
        x = self.fc(x[:, -1, :])
        return x

model = LSTMClassifier(embedding_dim, 64, len(set(labels)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

# Fuzzy Matching Function
def correct_spelling(text, choices):
    best_match, confidence = process.extractOne(text, choices)
    return best_match if confidence > 70 else "unknown_command"

# Predict function
def predict_command(text):
    corrected_text = correct_spelling(text, commands)
    if corrected_text == "unknown_command":
        return "unknown_command"
    vector = torch.tensor(text_to_vector(corrected_text), dtype=torch.float32)
    output = model(vector.unsqueeze(0))
    predicted_class = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]

# Load Wav2Vec2 Model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60")
model_stt = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60")

@app.post("/process_speech/")
async def process_speech(file: UploadFile = File(...)):
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    clip = AudioSegment.from_file(file_path)
    samples = np.array(clip.get_array_of_samples(), dtype=np.float32)
    samples = nr.reduce_noise(y=samples, sr=16000)
    inputs = processor(samples, sampling_rate=16000, return_tensors="pt", padding=True)
    logits = model_stt(inputs.input_values).logits
    tokens = torch.argmax(logits, axis=-1)
    text_output = processor.batch_decode(tokens)[0].lower()
    predicted_command = predict_command(text_output)
    os.remove(file_path)
    return {"text": text_output, "command": predicted_command}

class CommandRequest(BaseModel):
    text: str

@app.post("/iot/")
def send_to_iot(command: CommandRequest):
    return {"status": "Command sent to IoT device", "command": command.text}

@app.post("/tts/")
def text_to_speech(request: CommandRequest):
    return {"audio": "output.wav", "text": request.text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
