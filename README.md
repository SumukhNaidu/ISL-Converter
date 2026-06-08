# ISL Sign Language to Text & Speech Converter

Real-time Indian Sign Language (ISL) recognition system that converts 
hand gestures to text and spoken audio — fully offline, zero cost.

## Results
- Test accuracy: 98.44%
- Model size: 160.7 KB
- Inference: real-time on CPU
- Signs: 8 ISL signs

## Tech Stack
- Python, TensorFlow, Keras, MediaPipe, OpenCV
- LSTM neural network (2 layers)
- TFLite for on-device inference
- pyttsx3 / Windows SAPI for speech

## How it works
1. Webcam captures 30fps video
2. MediaPipe extracts 21 hand landmarks per frame
3. 30-frame sliding window fed into LSTM model
4. Model predicts ISL sign with confidence score
5. Recognised signs spoken aloud via TTS

## Setup
git clone https://github.com/SumukhNaidu/ISL-Converter.git
cd ISL-Converter
pip install -r requirements.txt
python app.py

## Signs Supported
HELLO, THANK YOU, WATER, FOOD, HELP, YES, NO, PLEASE

## Project Structure
data_collection/    — webcam recording scripts
training/           — LSTM model, preprocessing, TFLite conversion
app.py              — real-time recognition app with speech output

## Future Work
- Add 100+ ISL signs
- Flutter mobile app (Android)
- Sentence-level recognition
- Regional language speech output