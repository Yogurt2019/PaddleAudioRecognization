import os
import random
import librosa
import numpy as np

label_name_dict = {}

AUDIO_LEN = 13
MFCC_LEN = 30


def load_wav(wav_path):
    audio, sr = librosa.load(wav_path)
    processed_audio = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_LEN)
    x = np.zeros(shape=(MFCC_LEN, AUDIO_LEN))
    x[:, :AUDIO_LEN] = processed_audio
    x = np.array(x) / 100
    return x


def train_r(data_path='data'):
    def reader():
        data_label_dict = {}
        wav_list = []
        class_folder = os.path.join(data_path, 'train')
        classes = os.listdir(class_folder)
        label_num = 0
        for each_class in classes:
            wav_files = os.listdir(os.path.join(class_folder, each_class))
            wav_folder = os.path.join(class_folder, each_class)
            for wav_file in wav_files:
                wav_path = os.path.join(wav_folder, wav_file)
                label_name_dict[label_num] = each_class
                wav_list.append(wav_path)
                data_label_dict[wav_path] = label_num
        random.shuffle(wav_list)
        for wav_path in wav_list:
            x = load_wav(wav_path)
            label_num = data_label_dict[wav_path]
            yield x, label_num
    return reader


def test_r(data_path='data'):
    def reader():
        data_label_dict = {}
        wav_list = []
        class_folder = os.path.join(data_path, 'test')
        classes = os.listdir(class_folder)
        for each_class in classes:
            wav_files = os.listdir(os.path.join(class_folder, each_class))
            wav_folder = os.path.join(class_folder, each_class)
            for wav_file in wav_files:
                wav_path = os.path.join(wav_folder, wav_file)
                label_name_dict[label_num] = each_class
                wav_list.append(wav_path)
                data_label_dict[wav_path] = label_num
            random.shuffle(wav_list)
            for wav_path in wav_list:
                x = load_wav(wav_path)
                label_num = data_label_dict[wav_path]
                yield x, label_num
    return reader


def train():
    return train_r()


def test():
    return test_r()

