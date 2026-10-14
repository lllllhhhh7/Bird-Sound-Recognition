import os
import numpy as np
import soundfile as sf
import librosa
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

def highpass_filter(signal, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    y = filtfilt(b, a, signal)
    return y

def normalize_preemphasized_audio(signal):
    normalized_signal = librosa.util.normalize(signal)
    preemphasized_signal = librosa.effects.preemphasis(normalized_signal, coef=0.97)
    return preemphasized_signal

def calculate_short_time_energy(signal, sr=32000, frame_length=1024, hop_length=256):
    filtered_signal = highpass_filter(signal, cutoff=150, fs=sr)
    preemphasized_signal = normalize_preemphasized_audio(filtered_signal)
    energy = librosa.feature.rms(y=preemphasized_signal, frame_length=frame_length, hop_length=hop_length, center=False)
    return energy[0], preemphasized_signal

def not_silent(energy, threshold_db=-40):
    energy_db = 10 * np.log10(energy + 1e-10)
    return energy_db > threshold_db

def add_silence_to_meet_threshold(chunk, samplerate, min_chunk_threshold, split_time_seconds):
    duration = len(chunk) / samplerate

    if duration >= min_chunk_threshold:
        silence_duration_samples = int((split_time_seconds - duration) * samplerate)
        silence = np.zeros(silence_duration_samples)
        audio_data_with_silence = np.concatenate((chunk, silence), axis=None)
        return audio_data_with_silence
    else:
        return None

def process_audio_file(filename, root, output_subfolder, silent_subfolder, split_audio, split_time_seconds, min_chunk_threshold, allowed_formats, energy_threshold_db, sample_rate, eliminate_mute):
    base_name, ext = os.path.splitext(filename)
    if any(ext.lower() in f".{fmt.lower()}" for fmt in allowed_formats):
        audio_file = os.path.join(root, filename)
        y, sr = librosa.load(audio_file, sr=sample_rate)

        if split_audio == True and eliminate_mute == True:
            if len(y) < int(min_chunk_threshold * sr):
                print(f'音频 "{os.path.basename(audio_file)}" 持续时间太短，已删除')
            else:
                chunks = [y[i:i + int(split_time_seconds * sr)] for i in
                          range(0, len(y), int(split_time_seconds * sr))]
                last_chunk = add_silence_to_meet_threshold(chunks[-1], sr, min_chunk_threshold, split_time_seconds)

                if last_chunk is None:
                    chunks = chunks[:-1]
                    print(f'音频 "{os.path.basename(audio_file)}" 最后一段音频持续时间太短，已删除')
                else:
                    chunks[-1] = last_chunk
                    print(f'音频 "{os.path.basename(audio_file)}" 最后一段音频已填充静音！')

                for i, chunk in enumerate(chunks):
                    energy, preemphasized_signal = calculate_short_time_energy(chunk, sr=sr, frame_length=1024, hop_length=256)  # 计算帧的能源
                    bool = not_silent(energy, threshold_db=energy_threshold_db)
                    if bool.any():
                        output_chunk_path = os.path.join(output_subfolder, f"{base_name}_chunk{i + 1}{ext}")
                    else:
                        output_chunk_path = os.path.join(silent_subfolder, f"{base_name}_chunk{i + 1}{ext}")
                    sf.write(output_chunk_path, preemphasized_signal, sr)

        elif split_audio == True and eliminate_mute == False:
            if len(y) < int(min_chunk_threshold * sr):
                print(f'音频 "{os.path.basename(audio_file)}" 持续时间太短，已删除')
            else:
                chunks = [y[i:i + int(split_time_seconds * sr)] for i in
                          range(0, len(y), int(split_time_seconds * sr))]
                last_chunk = add_silence_to_meet_threshold(chunks[-1], sr, min_chunk_threshold, split_time_seconds)

                if last_chunk is None:
                    chunks = chunks[:-1]
                    print(f'音频 "{os.path.basename(audio_file)}" 最后一段音频持续时间太短，已删除')
                else:
                    chunks[-1] = last_chunk
                    print(f'音频 "{os.path.basename(audio_file)}" 最后一段音频已填充静音！')

                for i, chunk in enumerate(chunks):
                    _, preemphasized_signal = calculate_short_time_energy(chunk, sr=sr, frame_length=1024, hop_length=256)
                    output_chunk_path = os.path.join(output_subfolder, f"{base_name}_chunk{i + 1}{ext}")
                    sf.write(output_chunk_path, preemphasized_signal, sr)

        elif split_audio == False and eliminate_mute == True:
            energy, preemphasized_signal = calculate_short_time_energy(y, sr=sr, frame_length=1024, hop_length=256)
            bool = not_silent(energy, threshold_db=energy_threshold_db)
            if bool.any():
                output_chunk_path = os.path.join(output_subfolder, f"{base_name}{ext}")
            else:
                output_chunk_path = os.path.join(silent_subfolder, f"{base_name}{ext}")
            sf.write(output_chunk_path, preemphasized_signal, sr)

def process_directory(input_folder, output_folder, silent_output_dir, split_audio, split_time_seconds, min_chunk_threshold, allowed_formats, energy_threshold_db, sample_rate, eliminate_mute):
    start_time = time.time()
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    print(f'静默音频目录清空完毕')
    if os.path.exists(silent_output_dir):
        shutil.rmtree(silent_output_dir)
    os.makedirs(silent_output_dir)
    print(f'有效音频目录清空完毕')

    cpu_cores = os.cpu_count()
    max_workers = cpu_cores // 8
    print(f'电脑CPU线程数为：{cpu_cores}')
    print(f'处理数据的线程数：{max_workers}')

    if split_audio and eliminate_mute:
        print('\n程序执行功能：1.分割音频 2.剔除静默音频')
    elif split_audio == True and eliminate_mute == False:
        print('\n程序执行功能：分割音频')
    elif split_audio == False and eliminate_mute == True:
        print('\n程序执行功能：剔除静默音频')
    else:
        print('\n程序未执行任何功能，请重新设置变量 split_audio 和 eliminate_mute')
        exit()

    total_files = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, dirs, filenames in os.walk(input_folder):
            total_files += len(filenames)
            relative_path = os.path.relpath(root, input_folder)

            output_subfolder = os.path.join(output_folder, relative_path)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            silent_subfolder = os.path.join(silent_output_dir, relative_path)
            if not os.path.exists(silent_subfolder):
                os.makedirs(silent_subfolder)

            for filename in filenames:
                executor.submit(process_audio_file, filename, root, output_subfolder, silent_subfolder, split_audio, split_time_seconds, min_chunk_threshold, allowed_formats, energy_threshold_db, sample_rate, eliminate_mute)

    end_time = time.time()
    total_time = round(end_time - start_time)

    if split_audio or eliminate_mute:
        print(f"\n处理数据量：{total_files} 个")
        print(f"总处理时间：{total_time} 秒")


if __name__ == "__main__":
    original_fixed_path = r'E:\01_User\Dengyuxin\07_论文提交程序\05 Result\01_去噪—预处理_44100'
    valid_fixed_path    = r'E:\01_User\Dengyuxin\07_论文提交程序\05 Result\02_background去噪—预处理—有效音频2s_44100'
    silent_fixed_path   = r'E:\01_User\Dengyuxin\07_论文提交程序\05 Result\02_background去噪—预处理—静音音频2s_44100'
    aduio_dirs = ['00_北京百鸟科技数据库']
    split_audio = True
    split_time_seconds = 2
    min_chunk_threshold = 1
    allowed_formats = ['wav']
    sample_rate = 44100
    eliminate_mute = True
    energy_threshold_db = -12

    for aduio_dir in aduio_dirs:
        input_folder = os.path.join(original_fixed_path, aduio_dir)
        valid_output_folder = os.path.join(valid_fixed_path, aduio_dir)
        silent_output_dir = os.path.join(silent_fixed_path, aduio_dir)

        process_directory(input_folder, valid_output_folder, silent_output_dir, split_audio, split_time_seconds, min_chunk_threshold, allowed_formats, energy_threshold_db, sample_rate, eliminate_mute)