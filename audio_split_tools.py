from pydub.silence import *
import os
from pydub import AudioSegment


def split_and_export(dir):
    print('split and export mode')
    files = os.listdir(dir)
    print(files)
    for file in files:
        print(file)
        each_file = os.path.join(dir, file)
        seg1 = AudioSegment.from_wav(each_file)
        split_range = split_on_silence(seg1, min_silence_len=200, silence_thresh=-30, keep_silence=200, seek_step=1)
        i = 1
        for audio_seg in split_range:
            folder_path = os.path.join(dir, file.split('.')[0])
            if os.path.exists(folder_path) is not True:
                os.makedirs(folder_path)
            file1 = file.split('.')[0] + '-' + str(i) + '.wav'
            file_path = os.path.join(folder_path, file1)
            audio_seg.export(file_path, format="wav")
            print('Successfully exported {}'.format(file_path))
            i += 1


def split_equally_and_export(source_dir, target_dir):
    print('split equally and export mode')
    file_name_counter = 0
    folders = os.listdir(source_dir)
    for folder in folders:
        files = os.listdir(os.path.join(source_dir, folder))
        for file in files:
            each_file = os.path.join(source_dir, folder, file)
            print('source file:' + each_file)
            audio_seg = AudioSegment.from_wav(each_file)
            for seg in audio_seg[::300]:
                file_name_counter += 1
                file1 = file.split('.')[0] + '-' + str(file_name_counter) + '.wav'
                target_folder = os.path.join(target_dir, folder)
                if os.path.exists(target_folder) is not True:
                    os.mkdir(target_folder)
                file_path = os.path.join(target_folder, file1)
                print('target file:' + file_path)
                seg.export(file_path, format="wav")
                print('successfully exported {}'.format(file_path))
            file_name_counter = 0


def delete_silence_audio(dir):
    print('delete_silence_audio mode')
    deleted_file = []
    not_deleted_file = []
    files = os.listdir(dir)
    for file in files:
        each_file = os.path.join(dir, file)
        print(each_file)
        audio_seg = AudioSegment.from_wav(each_file)
        nonsilent_range = detect_nonsilent(audio_seg, min_silence_len=500, silence_thresh=-25, seek_step=5)
        if len(nonsilent_range) == 0:
            os.remove(each_file)
            print('deleted silence audio\t' + each_file)
            deleted_file.append(each_file)
        else:
            print(each_file + 'not deleted')
            not_deleted_file.append(each_file)
    print('deleted file list:')
    for i in range(len(deleted_file)):
        print(deleted_file[i])
    print('not deleted file list:')
    for i in range(len(not_deleted_file)):
        print(not_deleted_file[i])


def split_one_file_equally(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    seg = AudioSegment.from_wav(file_path)
    file_name_counter = 0
    for seg_clip in seg[::300]:
        file_name_counter += 1
        file1 = file_path.split('.')[0] + '-' + str(file_name_counter) + '.wav'
        out_folder_path = os.path.join(folder_path, file1)
        print('target file:' + out_folder_path)
        seg_clip.export(out_folder_path, format="wav")

s = r'E:\converted'
t = r'E:\converted_splited'
split_equally_and_export(s ,t)

