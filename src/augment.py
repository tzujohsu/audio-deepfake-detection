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
# def augment_real_audio(audio_data, labels, num_augments=1, sr=16000):
#     augmented_data = []
#     augmented_labels = []
#     MAX_SEQ_LENGTH = audio_data[0].shape[1]
#     print(audio_data.shape)
#     print(audio_data[0].shape)
#     print(MAX_SEQ_LENGTH)

#     for audio, label in tqdm(zip(audio_data, labels), total=len(labels), desc="Augmenting real audio"):
#         if label == 1:
#             augmented_data.append(audio)
#             augmented_labels.append(label)

#             for _ in range(num_augments):
#                 s = time()
#                 augmented = time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
                
#                 if len(augmented) >= MAX_SEQ_LENGTH:
#                     augmented = augmented[:MAX_SEQ_LENGTH]
#                 else:
#                     augmented = np.pad(augmented, (0, MAX_SEQ_LENGTH - len(augmented)))
#                 augmented_data.append(augmented)
#                 augmented_labels.append(label)
#                 print(f"Step 1 Done: {time() - s}")

#                 s = time()
#                 augmented = pitch_shift(audio, sr, n_steps=np.random.randint(-2, 3))
                
#                 if len(augmented) >= MAX_SEQ_LENGTH:
#                     augmented = augmented[:MAX_SEQ_LENGTH]
#                 else:
#                     augmented = np.pad(augmented, (0, MAX_SEQ_LENGTH - len(augmented)))
#                 augmented_data.append(augmented)
#                 augmented_labels.append(label)
#                 print(f"Step 2 Done: {time() - s}")

#                 s = time()


#                 augmented = add_noise(audio, noise_factor=np.random.uniform(0.001, 0.01))
#                 if len(augmented) >= MAX_SEQ_LENGTH:
#                     augmented = augmented[:MAX_SEQ_LENGTH]
#                 else:
#                     augmented = np.pad(augmented, (0, MAX_SEQ_LENGTH - len(augmented)))
#                 augmented_data.append(augmented)
#                 augmented_labels.append(label)
#                 print(f"Step 3 Done: {time() - s}")

#                 s = time()


#                 augmented = time_shift(audio, shift_max=np.random.uniform(0.1, 0.2))
#                 if len(augmented) >= MAX_SEQ_LENGTH:
#                     augmented = augmented[:MAX_SEQ_LENGTH]
#                 else:
#                     augmented = np.pad(augmented, (0, MAX_SEQ_LENGTH - len(augmented)))
#                 augmented_data.append(augmented)
#                 augmented_labels.append(label)
#                 print(f"Step 4 Done: {time() - s}")

#                 s = time()


#                 augmented = volume_scaling(audio, gain=np.random.uniform(0.7, 1.3))
#                 if len(augmented) >= MAX_SEQ_LENGTH:
#                     augmented = augmented[:MAX_SEQ_LENGTH]
#                 else:
#                     augmented = np.pad(augmented, (0, MAX_SEQ_LENGTH - len(augmented)))
#                 augmented_data.append(augmented)
#                 augmented_labels.append(label)
#                 print(f"Step 5 Done: {time() - s}")

                
#     return np.array(augmented_data), np.array(augmented_labels)


def _pad_augmented_audio(augmented, max_seq_length):
    if len(augmented) >= max_seq_length:
        augmented = augmented[:max_seq_length]
    else:
        augmented = np.pad(augmented, (0, max_seq_length - len(augmented)))
    return augmented


def load_audio_files(files):
    audio_data = []

    for file_name in tqdm(files, total=len(files)):
      file_path = os.path.join(DATASET_PATH, file_name + ".flac")

      # Load audio file using librosa
      audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
      if len(audio) < MAX_SEQ_LENGTH:
          audio = np.pad(audio, (0, MAX_SEQ_LENGTH - len(audio)))
      else:
          audio = audio[:MAX_SEQ_LENGTH]
      audio_data.append(audio)

    return np.array(audio_data)


import soundfile as sf

if __name__ == "__main__":
    augmented_folder = 'ASVspoof2019_LA_aug/flac'
    if not os.path.exists('./LA/'+augmented_folder):
        os.makedirs(augmented_folder)

    # Get all the real data from the training set
    LABEL_FILE_PATH = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    speakers = []; filenames = []; labels = []

    with open(LABEL_FILE_PATH, 'r') as label_file:
        lines = label_file.readlines()

    for line in lines:
        parts = line.strip().split()
        speakers.append(parts[0])
        filenames.append(parts[1])

        label = 1 if parts[-1] == "bonafide" else 0
        labels.append(label)

    audio_data = load_audio_files(filenames)

    # Tuple: (speaker, filename, audio data, label)
    data = list(zip(speakers, filenames, audio_data, labels))

    protocol_data = []

    # Iterate through all data
    for speaker, filename, audio, label in tqdm(data):
        #  Augment the real data
        if label == 1:
            augmented_data = _pad_augmented_audio(time_stretch(audio, rate=np.random.uniform(0.8, 1.2)), max_seq_length)
            sf.write(f"./LA/ASVspoof2019_LA_aug/flac/{filename}_aug_1", augmented_data)
            protocol_data.append([speaker, f"{filename}_aug_1", '-', '-', 'bonafide'])

            augmented_data = _pad_augmented_audio(pitch_shift(audio, sr, n_steps=np.random.randint(-2, 3)), max_seq_length)
            sf.write(f"./LA/ASVspoof2019_LA_aug/flac/{filename}_aug_2", augmented_data)
            protocol_data.append([speaker, f"{filename}_aug_2", '-', '-', 'bonafide'])

            augmented_data = _pad_augmented_audio(add_noise(audio, noise_factor=np.random.uniform(0.001, 0.01)), max_seq_length)
            sf.write(f"./LA/ASVspoof2019_LA_aug/flac/{filename}_aug_3", augmented_data)
            protocol_data.append([speaker, f"{filename}_aug_3", '-', '-', 'bonafide'])

            augmented_data = _pad_augmented_audio(time_shift(audio, shift_max=np.random.uniform(0.1, 0.2)), max_seq_length)
            sf.write(f"./LA/ASVspoof2019_LA_aug/flac/{filename}_aug_4", augmented_data)
            protocol_data.append([speaker, f"{filename}_aug_4", '-', '-', 'bonafide'])

            augmented_data = _pad_augmented_audio(volume_scaling(audio, gain=np.random.uniform(0.7, 1.3)), max_seq_length)
            sf.write(f"./LA/ASVspoof2019_LA_aug/flac/{filename}_aug_5", augmented_data)
            protocol_data.append([speaker, f"{filename}_aug_5", '-', '-', 'bonafide'])

            # original_file_name = ()
            # save it as f"{original_file_name}_aug_{i}"

    # create protocol file with columns -> speaker_id,utt_id,config,attacks,key
        # utt_id (file_name), key -> bonafide

    filename = 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019_aug.LA.cm.train.trn.txt'
    with open(filename, "w") as file:
        for row in protocol_data:
            file.write(' '.join(map(str, row)) + '\n')
