import os
import shutil
import soundfile as sf
import librosa
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

def copy_original_to_edit(input_folder, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    shutil.copytree(input_folder, output_folder)

def delete_non_audio_files(output_folder, audio_formats):
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            ext = os.path.splitext(file)[1][1:].lower()
            if ext not in audio_formats:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"删除 {file_path} 失败: {e}")

def move_audio_files(output_folder):
    directories = [os.path.join(output_folder, d) for d in os.listdir(output_folder) if
                   os.path.isdir(os.path.join(output_folder, d))]

    for root, dirs, files in os.walk(output_folder):
        for filename in files:
            ext = os.path.splitext(filename)[1][1:].lower()
            if ext in audio_formats:
                audio_file_path = os.path.join(root, filename)
                target_dir = next((d for d in directories if d in os.path.dirname(root)), None)
                if target_dir:
                    target_file_path = os.path.join(target_dir, filename)
                    shutil.move(audio_file_path, target_file_path)

def remove_empty_folders(output_folder):
    for foldername, subfolders, filenames in os.walk(output_folder, topdown=False):
        for subfolder in subfolders:
            subfolder_path = os.path.join(foldername, subfolder)
            for sub_subfolder, _, _ in os.walk(subfolder_path, topdown=False):
                if not os.listdir(sub_subfolder):
                    try:
                        os.rmdir(sub_subfolder)
                    except OSError as e:
                        print(f'未能删除空文件夹 {sub_subfolder}，原因: {e.strerror}')

        if not subfolders and not filenames:
            try:
                os.rmdir(foldername)
            except OSError as e:
                print(f'空文件夹删除失败：{foldername}，原因: {e.strerror}')


def convert_file_to_wav(audio_file, audio_formats, save_format, sample_rate):
    ext = os.path.splitext(audio_file)[1][1:].lower()
    if ext in audio_formats:
        output_file = os.path.splitext(audio_file)[0] + '.' + save_format

        try:
            probe_result = subprocess.check_output(
                ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of',
                 'default=noprint_wrappers=1:nokey=1', audio_file], text=True)
            if 'audio' not in probe_result.splitlines():
                print(f"警告：文件 {audio_file} 似乎不包含音频流，跳过转换.")
                os.remove(audio_file)

            signal, sample_rate = librosa.load(audio_file, sr=sample_rate)
            os.remove(audio_file)
            sf.write(output_file, signal, sample_rate)

        except FileNotFoundError:
            print("警告：未检测到FFmpeg，请确保已安装并在系统PATH中可访问，以便验证音频文件并转换。跳过文件 {} 的验证和转换。".format(audio_file))
        except subprocess.CalledProcessError as e:
            output_message = e.output if isinstance(e.output, str) else e.output.decode()
            print(f"错误：ffprobe 在处理文件 {audio_file} 时遇到问题，输出结果：{output_message}. 跳过转换.")
            os.remove(audio_file)
        except Exception as e:
            print(f"错误：在处理文件 {audio_file} 时发生异常，原因：{str(e)}，跳过转换.")
            os.remove(audio_file)

def convert_to_wav(output_folder, audio_formats, save_format, sample_rate, max_workers=4):
    total_files = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, dirs, files in os.walk(output_folder):
            total_files += len(files)
            for filename in files:
                audio_file = os.path.join(root, filename)
                executor.submit(convert_file_to_wav, audio_file, audio_formats, save_format, sample_rate)
    return total_files

def main(input_folder, output_folder, audio_formats, save_format, sample_rate):
    start_time = time.time()

    copy_original_to_edit(input_folder, output_folder)
    print("1.复制文件完成")
    delete_non_audio_files(output_folder, audio_formats)
    print("2.删除非音频文件完成")
    move_audio_files(output_folder)
    print("3.音频文件移动完成")
    remove_empty_folders(output_folder)
    print('4.删除空文件夹完成')

    cpu_cores = os.cpu_count()
    max_workers = -(-cpu_cores // 4)
    print(f'\n电脑CPU线程数为：{cpu_cores}')
    print(f'处理数据的线程数：{max_workers}')
    total_files = convert_to_wav(output_folder, audio_formats, save_format, sample_rate, max_workers=max_workers)
    print('5.转换音频文件为wav格式完成')

    end_time = time.time()
    total_time = round(end_time - start_time)
    print(f"\n处理数据量：{total_files} 个")
    print(f"总处理时间：{total_time} 秒")

if __name__ == '__main__':
    original_fixed_path = r'E:\01_User\Dengyuxin\01_dataset'
    edit_fixed_path     = r'E:\01_User\Dengyuxin\07_论文提交程序\05 Result\01_去噪—预处理_44100'

    aduio_dirs = ['00_北京百鸟科技数据库']
    audio_formats = ['mp3', 'wav', 'ogg']
    save_format = 'wav'
    sample_rate = 44100

    for aduio_dir in aduio_dirs:
        input_folder = os.path.join(original_fixed_path, aduio_dir)
        output_folder = os.path.join(edit_fixed_path, aduio_dir)

        main(input_folder, output_folder, audio_formats, save_format, sample_rate)


