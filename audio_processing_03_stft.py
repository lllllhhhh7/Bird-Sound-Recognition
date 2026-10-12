import os
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from PIL import Image
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

def save_spectrum_image(audio_path, output_path, sample_rate, size):
    signal, sample_rate = librosa.load(audio_path, sr=sample_rate)

    stft = librosa.stft(
        y=signal, n_fft=1024, hop_length=256,
        window='hann'
    )
    magnitude_spec = np.abs(stft)
    stft_db = librosa.amplitude_to_db(magnitude_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=size)

    librosa.display.specshow(stft_db, sr=sample_rate, hop_length=256, ax=ax, cmap='magma')
    ax.axis('off')
    fig.tight_layout(pad=0.0)

    plt.savefig(output_path, format='png')
    img = Image.open(output_path).convert('RGB')
    img.save(output_path, 'PNG')

    plt.ioff()
    plt.close(fig)

def process_birdsongs_folder(input_folder, output_folder, allowed_formats, sample_rate, continuation):

    start_time = time.time()

    if continuation is False:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        print('输出目录清空完成，开始生成STFT语谱图...\n')
    else:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print('输出目录创建完成，开始生成STFT语谱图...\n')

    total_files = 0
    cpu_cores = os.cpu_count()
    max_workers = -(-cpu_cores // 8)
    print(f'电脑CPU进程数：{cpu_cores}')
    print(f'程序使用进程数：{max_workers}\n')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for root, dirs, files in os.walk(input_folder):
            total_files += len(files)
            for file_name in files:
                _, ext = os.path.splitext(file_name)
                if ext[1:].lower() in allowed_formats:
                    audio_path = os.path.join(root, file_name)
                    base_name = os.path.splitext(file_name)[0]
                    output_spec_path = os.path.join(output_folder, f"{base_name}_stft.png")

                    if not os.path.exists(output_spec_path):
                        future = executor.submit(
                            save_spectrum_image,
                            audio_path, output_spec_path,
                            sample_rate, (3.84, 3.84)
                        )
                        futures.append(future)
                    else:
                        print(f"{output_spec_path} 已存在，跳过")

        for future in as_completed(futures):
            future.result()

    end_time = time.time()
    total_time = round(end_time - start_time)
    print(f"处理数据量：{total_files} 个")
    print(f"总处理时间：{total_time} 秒")


if __name__ == "__main__":
    original_fixed_path = r'E:\01_User\Dengyuxin\07_论文提交程序\05 Result\02_background去噪—预处理—有效音频2s_44100\00_北京百鸟科技数据库'
    edit_fixed_path = r'E:\01_User\Dengyuxin\07_论文提交程序\05 Result\03_去噪—预处理—分割2s—有效音频_STFT语谱图_44100_384'
    aduio_dirs = ['Accipiter gentilis_苍鹰', 'Anas crecca_绿翅鸭', 'Anas platyrhynchos_绿头鸭', 'Anser anser_灰雁',
                  'Ardea cinerea_苍鹭', 'Buteo buteo_欧亚鵟', 'Coturnix coturnix_西鹌鹑', 'Cygnus cygnus_大天鹅',
                  'Fulica atra_白骨顶', 'Gavia stellata_红喉潜鸟', 'Himantopus himantopus_黑翅长脚鹬',
                  'Passer domesticus_麻雀', 'Perdix perdix_灰山鹑', 'Phalacrocorax carbo_普通鸬鹚',
                  'Phasianus colchicus_雉鸡', 'Rallus aquaticus_西方秧鸡', 'Tringa glareola_林鹬',
                  'Tringa ochropus_白腰草鹬', 'Tringa totanus_红脚鹬', 'Vanellus vanellus_凤头麦鸡']

    allowed_formats = ['wav']
    sample_rate = 44100
    continuation = False

    for aduio_dir in aduio_dirs:
        input_folder = os.path.join(original_fixed_path, aduio_dir)
        output_folder = os.path.join(edit_fixed_path, aduio_dir)

        if not os.path.exists(input_folder):
            print(f"警告：输入路径 {input_folder} 不存在，跳过处理")
            continue

        print(f"\n===== 开始处理 {aduio_dir} =====")
        process_birdsongs_folder(input_folder, output_folder, allowed_formats, sample_rate, continuation)

    print("\n所有文件夹处理完成！")