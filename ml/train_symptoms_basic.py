"""
Basic symptom severity classifier training script.
Outputs: text_classifier.h5, text_classifier_labels.txt, vocab.txt
"""
import os
import re
import string
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_CSV = os.path.join('..', 'datasets', 'DiseaseAndSymptoms.csv')
ASSETS_DIR = os.path.join('..', 'app', 'src', 'main', 'assets', 'models')
os.makedirs(ASSETS_DIR, exist_ok=True)

def clean(t):
    t = t.lower().replace('_', ' ')
    t = t.translate(str.maketrans('', '', string.punctuation))
    t = re.sub(r'\s+', ' ', t).strip()
    return t

df = pd.read_csv(DATA_CSV)
sym_cols = [c for c in df.columns if c.lower().startswith('symptom')]

texts, labels = [], []
for _, row in df.iterrows():
    toks = [str(row[c]).strip() for c in sym_cols if pd.notna(row[c]) and str(row[c]).strip()]
    if not toks:
        continue
    texts.append(clean(' '.join(toks)))
    labels.append(row['Severity'] if 'Severity' in df.columns else 'Moderate')

if 'Severity' not in df.columns:
    # simple heuristic
    sev_k = ['emergency','severe','chest pain','breathlessness','unconscious']
    mild_k = ['mild','itching','rash','dry']
    new_labels = []
    for t in texts:
        s = t.lower()
        if any(k in s for k in sev_k): new_labels.append('Severe')
        elif any(k in s for k in mild_k): new_labels.append('Mild')
        else: new_labels.append('Moderate')
    labels = new_labels

le = LabelEncoder()
y = le.fit_transform(labels)
y = tf.keras.utils.to_categorical(y)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=128, padding='post')

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = models.Sequential([
    layers.Embedding(10000, 128, input_length=128),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(y.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=10, batch_size=32)

keras_path = os.path.join(ASSETS_DIR, 'text_classifier.h5')
model.save(keras_path)
labels_path = os.path.join(ASSETS_DIR, 'text_classifier_labels.txt')
with open(labels_path, 'w', encoding='utf-8') as f:
    for c in le.classes_:
        f.write(f"{c}\n")

vocab_path = os.path.join(ASSETS_DIR, 'vocab.txt')
with open(vocab_path, 'w', encoding='utf-8') as f:
    # Ensure <PAD> and <OOV> exist at top for app expectations
    f.write('<PAD>\n')
    f.write('<OOV>\n')
    for word, idx in sorted(tokenizer.word_index.items(), key=lambda x: x[1]):
        if word in {'<PAD>','<OOV>'}:
            continue
        f.write(f"{word}\n")

print(f"Saved -> {keras_path}, {labels_path}, {vocab_path}")
