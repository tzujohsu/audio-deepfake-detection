from time import time

import numpy as np
import librosa
from tqdm import tqdm

#%% Augmentations
def time_stretch(audio, rate=1.0):
    return librosa.effects.time_stretch(y=audio, rate=rate)

def pitch_shift(audio, sr, n_steps=0):
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def time_shift(audio, shift_max=0.2):
    shift = np.random.randint(len(audio) * -shift_max, len(audio) * shift_max)
    return np.roll(audio, shift)

def volume_scaling(audio, gain=1.0):
    return audio * gain

#%% Augment Data
def augment_real_audio(audio_data, labels, num_augments=1, sr=16000):
    augmented_data = []
    augmented_labels = []
    MAX_SEQ_LENGTH = audio_data[0].shape[1]
    print(audio_data.shape)
    print(audio_data[0].shape)
    print(MAX_SEQ_LENGTH)

    for audio, label in tqdm(zip(audio_data, labels), total=len(labels), desc="Augmenting real audio"):
        if label == 1:
            augmented_data.append(audio)
            augmented_labels.append(label)

            for _ in range(num_augments):
                s = time()
                augmented = time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
                
                if len(augmented) >= MAX_SEQ_LENGTH:
                    augmented = augmented[:MAX_SEQ_LENGTH]
                else:
                    augmented = np.pad(augmented, (0, MAX_SEQ_LENGTH - len(augmented)))
                augmented_data.append(augmented)
                augmented_labels.append(label)
                print(f"Step 1 Done: {time() - s}")

                s = time()
                augmented = pitch_shift(audio, sr, n_steps=np.random.randint(-2, 3))
                
                if len(augmented) >= MAX_SEQ_LENGTH:
                    augmented = augmented[:MAX_SEQ_LENGTH]
                else:
                    augmented = np.pad(augmented, (0, MAX_SEQ_LENGTH - len(augmented)))
                augmented_data.append(augmented)
                augmented_labels.append(label)
                print(f"Step 2 Done: {time() - s}")

                s = time()


                augmented = add_noise(audio, noise_factor=np.random.uniform(0.001, 0.01))
                if len(augmented) >= MAX_SEQ_LENGTH:
                    augmented = augmented[:MAX_SEQ_LENGTH]
                else:
                    augmented = np.pad(augmented, (0, MAX_SEQ_LENGTH - len(augmented)))
                augmented_data.append(augmented)
                augmented_labels.append(label)
                print(f"Step 3 Done: {time() - s}")

                s = time()


                augmented = time_shift(audio, shift_max=np.random.uniform(0.1, 0.2))
                if len(augmented) >= MAX_SEQ_LENGTH:
                    augmented = augmented[:MAX_SEQ_LENGTH]
                else:
                    augmented = np.pad(augmented, (0, MAX_SEQ_LENGTH - len(augmented)))
                augmented_data.append(augmented)
                augmented_labels.append(label)
                print(f"Step 4 Done: {time() - s}")

                s = time()


                augmented = volume_scaling(audio, gain=np.random.uniform(0.7, 1.3))
                if len(augmented) >= MAX_SEQ_LENGTH:
                    augmented = augmented[:MAX_SEQ_LENGTH]
                else:
                    augmented = np.pad(augmented, (0, MAX_SEQ_LENGTH - len(augmented)))
                augmented_data.append(augmented)
                augmented_labels.append(label)
                print(f"Step 5 Done: {time() - s}")

                
    return np.array(augmented_data), np.array(augmented_labels)

