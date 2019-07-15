from pydub import AudioSegment
import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import pickle
import random


AUDIO_LEN = 30
MFCC_LEN = 13

def create_data():
    item_len = 2000

    base_path = "./all/"

    for file in os.listdir(base_path):
        real_path = base_path + file
        audio = AudioSegment.from_file(real_path)
        audio_len = len(audio)

        steps = audio_len // item_len
        for step in range(max(1, steps)):
            item_audio = audio[step * item_len: (step + 1) * item_len]
            save_audio = item_audio + AudioSegment.silent(item_len - len(item_audio))

            save_audio.export("./data/" + str(step) + file, format="wav")

# create_data()
# exit()

def split_train_test():
    base_dir = "./data/"
    dogs = []
    cats = []
    for file in os.listdir(base_dir):
        real_file = base_dir + file
        if "cat" in file:
            cats.append(real_file)
        else:
            dogs.append(real_file)
    test = dogs[:10] + cats[:10]
    train = dogs[10:] + cats[10:]
    random.shuffle(train)
    random.shuffle(test)
    pickle.dump(train, open("./train", "wb"))
    pickle.dump(test, open("./test", "wb"))

train_data = pickle.load(open("train", "rb"))
test_data = pickle.load(open("test", "rb"))

def get_train_or_test(type, batch_size = 30):
    x = []
    y = []
    data = train_data if type == "train" else test_data
    if type == "test":
        batch_size = 10
    all_dogs = [item for item in data if "dog" in item]
    all_cats = [item for item in data if "cat" in item]

    sample_dogs = random.sample(all_dogs, int(batch_size / 2))
    sample_cats = random.sample(all_cats, batch_size - int(batch_size / 2))

    sample = sample_dogs + sample_cats
    random.shuffle(sample)

    for item in sample:
        try:
            fs, audio = wav.read(item)
            processed_audio = mfcc(audio, samplerate=fs)
            x_hold = np.zeros(shape=(AUDIO_LEN, MFCC_LEN))
            x_hold[:len(processed_audio), :] = processed_audio
            x.append(x_hold)
            if type == "train":
                reverse = []
                for data in x_hold:
                    list(data).reverse()
                    reverse.append(data)
                x.append(reverse)
                y.append([1, 0] if "cat" in item else [0, 1])
            y.append([1, 0] if "cat" in item else [0, 1])
        except:
            print("error")
            pass
    x = np.array(x) / 100

    x = np.array(x)
    y = np.array(y)
    return x, y
