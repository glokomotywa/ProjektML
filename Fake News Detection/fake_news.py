import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D,
                                     Flatten, Embedding, LSTM, BatchNormalization,
                                     GlobalMaxPooling1D, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


# Load datasets
def load_datasets():
    columns = ["id", "label", "statement", "subject", "speaker", "job_title",
               "state_info", "party_affiliation", "barely_true_counts",
               "false_counts", "half_true_counts", "mostly_true_counts",
               "pants_on_fire_counts", "context"]
    train_df = pd.read_csv("train.tsv", sep="\t", header=None, names=columns)
    test_df = pd.read_csv("test.tsv", sep="\t", header=None, names=columns)
    valid_df = pd.read_csv("valid.tsv", sep="\t", header=None, names=columns)
    return train_df, test_df, valid_df


train_df, test_df, valid_df = load_datasets()
full_train_df = pd.concat([train_df, valid_df]).sample(frac=1, random_state=42)

# Split into train and validation sets
train_df_split, val_df_split = train_test_split(
    full_train_df, test_size=0.2, random_state=42, stratify=full_train_df["label"]
)

# Label encoding
label_mapping = {
    "pants-fire": 0, "false": 0, "barely-true": 0,
    "half-true": 1, "mostly-true": 1, "true": 1
}
y_train = train_df_split["label"].map(label_mapping).values
y_val = val_df_split["label"].map(label_mapping).values
y_test = test_df["label"].map(label_mapping).values

# Convert to categorical
num_classes = 2
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Text preprocessing parameters
MAX_WORDS = 10000
MAX_LEN = 300
EMBEDDING_DIM = 300  # For GloVe embeddings

# Tokenization for LSTM and CNN
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df_split["statement"])

# Sequences for LSTM/CNN
train_sequences = tokenizer.texts_to_sequences(train_df_split["statement"])
val_sequences = tokenizer.texts_to_sequences(val_df_split["statement"])
test_sequences = tokenizer.texts_to_sequences(test_df["statement"])

X_train_seq = pad_sequences(train_sequences, maxlen=MAX_LEN)
X_val_seq = pad_sequences(val_sequences, maxlen=MAX_LEN)
X_test_seq = pad_sequences(test_sequences, maxlen=MAX_LEN)

# TF-IDF for MLP
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(train_df_split["statement"])
X_val_tfidf = vectorizer.transform(val_df_split["statement"])
X_test_tfidf = vectorizer.transform(test_df["statement"])

# Load GloVe embeddings 
embeddings_index = {}
with open('glove.6B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < MAX_WORDS:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Model architectures
models = {
    "MLP": Sequential([
        Dense(512, activation='relu', kernel_regularizer=l2(0.01),
              input_shape=(X_train_tfidf.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ]),

    "CNN": Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LEN,
                  trainable=False),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(2),
        Conv1D(256, 3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ]),

    "BiLSTM": Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LEN,
                  trainable=False),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)),
        Bidirectional(LSTM(64, dropout=0.2)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
}

# Compile and train models
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history_dict = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Select input data
    if name == "MLP":
        X_train, X_val = X_train_tfidf.toarray(), X_val_tfidf.toarray()
    else:
        X_train, X_val = X_train_seq, X_val_seq

    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val_cat),
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )
    history_dict[name] = history

    # Evaluate on test set
    if name == "MLP":
        X_test = X_test_tfidf.toarray()
    else:
        X_test = X_test_seq

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(f"\n{name} Test Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=["FALSE", "TRUE"]))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_classes))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["FALSE", "TRUE"],
                yticklabels=["FALSE", "TRUE"])
    plt.title(f'{name} Confusion Matrix')
    plt.show()

# Plot learning curves
for name, history in history_dict.items():
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{name} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{name} Loss')
    plt.legend()
    plt.show()