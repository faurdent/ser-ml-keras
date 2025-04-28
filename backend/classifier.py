import keras
import numpy as np
from keras import layers, Model, Sequential
import librosa
import librosa.feature
from pydub import AudioSegment
import warnings

class BaseModel(Model):
  def __init__(self, **kwargs):
      super().__init__()
      self.seq = Sequential()

  def call(self, inputs):
      return self.seq(inputs)

  def build(self, input_shape):
      self.seq.build()


@keras.saving.register_keras_serializable()
class LSTM3(BaseModel):
    def __init__(self, activation, **kwargs):
        super().__init__( **kwargs)
        self.activation = activation
        self.seq = Sequential([
            layers.Input(shape=(352, 15)),
            layers.Bidirectional(layers.LSTM(128, activation, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(64, activation)),
            layers.Dropout(0.4),
            layers.Dense(6, activation="softmax")
        ])

    def get_config(self):
      return {"activation": self.activation}


model = keras.models.load_model("./best_lstm.keras")

warnings.filterwarnings('ignore')

class SpeechEmotionClassifier:
    def __init__(self, model_path='best_lstm.keras'):
        self.model = keras.models.load_model(model_path)
        self.time_steps = 352
        self.emotion_map = {
            0: 'neutral',
            1: 'happy',
            2: 'sad',
            3: 'angry',
            4: 'fear',
            5: 'disgust'
        }

    def extract_features(self, audio_path):
        """Extract and normalize audio features matching the original 15 features."""
        try:
            # Load audio file
            _, sr = librosa.load(audio_path, duration=3)
            raw_audio = AudioSegment.from_wav(audio_path)
            samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')

            trimmed, _ = librosa.effects.trim(samples, top_db=25)
            if len(trimmed) < 180000:
                y = np.pad(trimmed, (0, 180000 - len(trimmed)), 'constant')
            else:
                y = trimmed[:180000]
            # Set consistent parameters
            frame_length = 2048
            hop_length = 512

            # Extract only the original features
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

            # Ensure all features have the same time steps
            min_time_steps = min(zcr.shape[1], rms.shape[1], mfccs.shape[1])

            # Trim all features to minimum length
            zcr = zcr[:, :min_time_steps]
            rms = rms[:, :min_time_steps]
            mfccs = mfccs[:, :min_time_steps]

            # Stack features (15 features total: 1 ZCR + 1 RMS + 13 MFCCs)
            features = np.vstack([zcr, rms, mfccs])
            return np.expand_dims(features.T, axis=0)

        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            raise

    def predict_emotion(self, audio_path):
        """Predict emotion with detailed probability output."""
        try:
            x_input = self.extract_features(audio_path)

            # Get predictions
            predictions = self.model.predict(x_input, verbose=0)
            # print(max(predictions[0]))
            predicted_class = np.argmax(predictions[0])
            # print(np.argmax(predictions, axis=1))
            predicted_emotion = self.emotion_map[predicted_class]

            return predicted_emotion, float(max(predictions[0]))

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None

