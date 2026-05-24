# =========================================================
# CUSTOMER SUPPORT TICKET CLASSIFICATION USING CNN
# =========================================================

# Install if needed:
# pip install tensorflow scikit-learn pandas numpy

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical


# =========================================================
# LOAD DATA
# =========================================================

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# =========================================================
# COMBINE TEXT COLUMNS
# =========================================================
# We combine subject + message because both contain text information

train_df["text"] = (
    train_df["subject"].astype(str) + " " +
    train_df["message"].astype(str)
)

test_df["text"] = (
    test_df["subject"].astype(str) + " " +
    test_df["message"].astype(str)
)

# =========================================================
# INPUT AND TARGET
# =========================================================

X = train_df["text"]

y = train_df["ticket_category"]

# =========================================================
# LABEL ENCODING TARGET VARIABLE
# =========================================================
# CNN works with numbers, not text labels

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# Convert into one-hot vectors
y_categorical = to_categorical(y_encoded)

# =========================================================
# TRAIN VALIDATION SPLIT
# =========================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================================================
# TOKENIZATION
# =========================================================
# Convert words into integer tokens

max_words = 20000

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(X_train)

# Convert text into sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(test_df["text"])

# =========================================================
# PADDING
# =========================================================
# Make all sequences same length

max_len = 150

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# =========================================================
# BUILD CNN MODEL
# =========================================================

model = Sequential()

# Embedding Layer
model.add(
    Embedding(
        input_dim=max_words,
        output_dim=128,
        input_length=max_len
    )
)

# CNN Layer
model.add(
    Conv1D(
        filters=128,
        kernel_size=5,
        activation='relu'
    )
)

# Pooling Layer
model.add(GlobalMaxPooling1D())

# Dense Layers
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

# Output Layer
model.add(
    Dense(
        y_categorical.shape[1],
        activation='softmax'
    )
)

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================================
# MODEL SUMMARY
# =========================================================

model.summary()

# =========================================================
# TRAIN MODEL
# =========================================================

history = model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_val_pad, y_val)
)

# =========================================================
# VALIDATION PREDICTIONS
# =========================================================

val_pred_probs = model.predict(X_val_pad)

# Convert probabilities into class index
val_pred = np.argmax(val_pred_probs, axis=1)

# Actual validation labels
y_val_actual = np.argmax(y_val, axis=1)

# =========================================================
# EVALUATION METRICS
# =========================================================

accuracy = accuracy_score(y_val_actual, val_pred)

macro_f1 = f1_score(
    y_val_actual,
    val_pred,
    average='macro'
)

weighted_f1 = f1_score(
    y_val_actual,
    val_pred,
    average='weighted'
)

# Final competition score
final_score = (
    0.4 * accuracy +
    0.3 * macro_f1 +
    0.3 * weighted_f1
) * 100

print("\nAccuracy :", accuracy)
print("Macro F1 :", macro_f1)
print("Weighted F1 :", weighted_f1)
print("Final Score :", final_score)

# =========================================================
# TEST PREDICTIONS
# =========================================================

test_pred_probs = model.predict(X_test_pad)

test_pred = np.argmax(test_pred_probs, axis=1)

# Convert numeric labels back to original labels
test_labels = label_encoder.inverse_transform(test_pred)

# =========================================================
# CREATE SUBMISSION FILE
# =========================================================

submission = pd.DataFrame({
    "ticket_id": test_df["ticket_id"],
    "ticket_category": test_labels
})

submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv file created successfully!")
