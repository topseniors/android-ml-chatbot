Includes:

 - Python training script
 - intents.json
 - TensorFlow Lite export
 - Android Kotlin app
 - Offline chatbot UI
 - Intent-based ML replies

Run training first:

```
cd training
pip install -r requirements.txt
python train.py
```

Then copy:

```
intent_model.tflite
metadata.json
```

into:

```
android/app/src/main/assets/
```

Open `android/` in Android Studio and run the app.
