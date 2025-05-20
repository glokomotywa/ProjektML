import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, LSTM, Embedding, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


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

# Label encoding
label_mapping = {
    "pants-fire": 0, "false": 0, "barely-true": 0,
    "half-true": 1, "mostly-true": 1, "true": 1
}
y_train = train_df["label"].map(label_mapping).values
y_val = valid_df["label"].map(label_mapping).values
y_test = test_df["label"].map(label_mapping).values

# Convert to categorical
num_classes = 2
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Text preprocessing
MAX_WORDS = 10000
MAX_LEN = 300

# Tokenization
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["statement"])

# Sequences
train_sequences = tokenizer.texts_to_sequences(train_df["statement"])
X_train_seq = pad_sequences(train_sequences, maxlen=MAX_LEN)
X_val_seq = pad_sequences(tokenizer.texts_to_sequences(valid_df["statement"]), maxlen=MAX_LEN)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(test_df["statement"]), maxlen=MAX_LEN)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(train_df["statement"])
X_val_tfidf = vectorizer.transform(valid_df["statement"])
X_test_tfidf = vectorizer.transform(test_df["statement"])

# Class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Model architectures
models = {
    "MLP": Sequential([
        Dense(256, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ]),

    "CNN": Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(num_classes, activation='softmax')
    ]),

    "BiLSTM": Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        Bidirectional(LSTM(64)),
        Dense(num_classes, activation='softmax')
    ])
}

history_dict = {}

# Training and evaluation
for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Select input data
    if name == "MLP":
        X_train, X_val = X_train_tfidf.toarray(), X_val_tfidf.toarray()
    else:
        X_train, X_val = X_train_seq, X_val_seq

    history_dict[name] = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val_cat),
        class_weight=class_weights,
        callbacks=[EarlyStopping(patience=2)],
        verbose=1
    )

    # Evaluation
    X_test = X_test_tfidf.toarray() if name == "MLP" else X_test_seq
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"\n{name} Test Report:")
    print(classification_report(y_test, y_pred_classes))
    print("Accuracy:", accuracy_score(y_test, y_pred_classes))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.show()

# Learning curves
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