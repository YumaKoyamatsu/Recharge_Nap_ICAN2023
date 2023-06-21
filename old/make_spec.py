from PIL import Image
import os
import concurrent.futures
import glob
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def wav2spec(sig, n_fft=16, hop_length=1, sr=1):
    # STFTを用いてスペクトログラムを計算
    D = np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=hop_length))
    # 対数スペクトログラムを計算
    D = librosa.amplitude_to_db(D, ref=np.max)
    # 画像として表示
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.axis('off')  # 軸の数値を表示しない
    #余白削除
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    # フィギュアのRGBデータを取得しPillowのImageに変換
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), data)
    # リサイズ
    img_resized = image.resize((224, 224))
    plt.close()
    img_np = np.array(img_resized)
    
    return img_np

def process_image(file, base_name):
    sig = []
    annot = []
    file_name = file.split("\\")[-1]
    #print(file_name)
    
    data = np.load(file, allow_pickle=True).item()
    sig.append(data["sig"])
    annot.append(data["annot"])
    
    sig = np.vstack(sig)
    annot = np.concatenate(annot)

    spec = []
    for i in range(sig.shape[0]):
        spec.append(wav2spec(sig[i]))
    
    sig = np.stack(spec)
    out = {"sig": sig,
        "annot": annot}
    out_name = f"{base_name}\\spec_{file_name}"
    np.save(out_name, out)

if __name__ == "__main__":
    base_name = "dataset\\YSYW_seq\\spec_RRI"
    os.makedirs(base_name, exist_ok=True)

    files = glob.glob("dataset\\YSYW_seq\\RRI\\*.npy")

    # 使用するスレッド数を指定
    num_threads = 24

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        # 各画像に対して処理を並列化
        for file in files:
            # スレッドを使用して画像処理と保存を実行
            executor.submit(process_image, file, base_name)
    # for file in files:
    #     process_image(file=file, base_name=base_name)

    print("処理が完了しました。")
