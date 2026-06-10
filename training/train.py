import json
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras

MAX_WORDS = 1000
MAX_LEN = 12
EPOCHS = 300


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []
tag_to_response = {}

for intent in data["intents"]:
    tag = intent["tag"]
    tag_to_response[tag] = intent["responses"]
    for pattern in intent["patterns"]:
        texts.append(clean_text(pattern))
        labels.append(tag)

tags = sorted(list(set(labels)))
tag_to_id = {tag: i for i, tag in enumerate(tags)}
y = np.array([tag_to_id[label] for label in labels])

vectorizer = keras.layers.TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode="int",
    output_sequence_length=MAX_LEN,
    standardize=None,
)
vectorizer.adapt(texts)

model = keras.Sequential([
    keras.Input(shape=(1,), dtype=tf.string),
    vectorizer,
    keras.layers.Embedding(MAX_WORDS, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(len(tags), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(np.array(texts), y, epochs=EPOCHS, verbose=0)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()
with open("intent_model.tflite", "wb") as f:
    f.write(tflite_model)

metadata = {
    "tags": tags,
    "responses": tag_to_response,
    "confidence_threshold": 0.55
}

with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Generated intent_model.tflite and metadata.json")
