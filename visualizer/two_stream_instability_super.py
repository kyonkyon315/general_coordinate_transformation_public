import struct
import numpy as np
import cv2
import glob
import imageio
import os
from concurrent.futures import ProcessPoolExecutor

# =============================
#   描画パラメータ (定数として定義)
# =============================
XMIN, XMAX = 0, 422.4
YMIN, YMAX = -16.5, 16.5
RES = 400

def world_to_img(x, y):
    ix = int((x - XMIN) / (XMAX - XMIN) * RES)
    iy = int((YMAX - y) / (YMAX - YMIN) * RES)
    return (ix, iy)

def process_frame(fn):
    """
    1つのファイルを読み込み、画像配列を生成して返す関数
    """
    try:
        with open(fn, "rb") as f:
            Nx = struct.unpack("q", f.read(8))[0]
            Ny = struct.unpack("q", f.read(8))[0]

            vx_all = np.zeros((Nx, Ny, 4))
            vy_all = np.zeros((Nx, Ny, 4))
            val_all = np.zeros((Nx, Ny))

            for i in range(Nx):
                for j in range(Ny):
                    vx_all[i,j] = struct.unpack("4d", f.read(32))
                    vy_all[i,j] = struct.unpack("4d", f.read(32))
                    val_all[i,j] = struct.unpack("d",  f.read(8))[0]

        # 画像化処理
        vmin, vmax = val_all.min(), val_all.max()
        val_norm = (val_all - vmin) / (vmax - vmin + 1e-12) * 255
        
        img = np.zeros((RES, RES, 3), dtype=np.uint8)
        for i in range(Nx):
            for j in range(Ny):
                pts = np.array([
                    world_to_img(vx_all[i,j,k], vy_all[i,j,k])
                    for k in range(4)
                ], np.int32)
                color = int(val_norm[i,j])
                cv2.fillPoly(img, [pts], (0, 0, color))
        
        print(f"Finished: {fn}",flush=True)
        return img
    except Exception as e:
        print(f"Error in {fn}: {e}")
        return None

# =============================
#   メイン処理
# =============================
if __name__ == "__main__":
    # ファイルリストの取得とフィルタリング
    #all_files = sorted(glob.glob("../output/two_stream/rank*.bin"), 
    #                   key=lambda s: int(s.split('/')[-1].split('.')[0]))
    for rank in [22,26,30]:
        all_files = ["../output/two_stream/rank_"+str(rank)+"__step_" + str(i) + ".bin" for i in range(4000)]
        
        # 条件: i%4 != 0 かつ i < 5000 のファイルのみ抽出
        target_files = [fn for i, fn in enumerate(all_files) if i % 4 == 0 and i < 5000]

        print(f"Total frames to process: {len(target_files)}",flush=True)

        # プロセスプールを使用して並列実行
        # executor.map は元のリストの順序を維持して結果を返します
        nproc = int(os.environ.get("NPROC", 8))
        print(f"Using {nproc} processes",flush=True)

        with ProcessPoolExecutor(max_workers=nproc) as executor:
            frames = list(executor.map(process_frame, target_files))

        # None（エラー）を除外
        frames = [f for f in frames if f is not None]

        # GIF 保存
        gif_path = "../output/two_stream_instability_"+str(rank)+".gif"
        fps = 15
        imageio.mimsave(gif_path, frames, fps=fps)

        print(f"GIF saved to {gif_path}")
