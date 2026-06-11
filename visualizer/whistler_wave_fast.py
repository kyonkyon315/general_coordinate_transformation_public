import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import gc # ガベージコレクション用

if __name__ == "__main__":

    # ファイル一覧取得
    files = glob.glob("../output/whistler/Ez_t_blockid_z_*.dat")

    # block_id でソート
    def extract_block_id(fname):
        return int(re.search(r'blockid_z_(\d+)', fname).group(1))

    files.sort(key=extract_block_id)

    # ======================
    # 対策①: float32 で読み込んでメモリ半減
    # ======================
    print("Loading data...")
    data_blocks = [np.loadtxt(f) for f in files]

    # 横方向に結合（z方向に並べる）
    Ez_global = np.hstack(data_blocks)
    
    # ======================
    # 対策②: 結合し終わったリストは即座にメモリ解放
    # ======================
    del data_blocks
    gc.collect() 
    print(f"Original shape: {Ez_global.shape}")

    # ======================
    # 物理パラメータ
    # ======================
    c = 3e8
    omega_pe = 564102.5
    omega_ce = omega_pe / 4.
    v_th = 0.003 * c  # ※シミュレーションで変更した場合はここも合わせる
    R_ce = v_th / omega_ce

    dx = 0.125 * R_ce
    dt_original = 0.01 / omega_ce

    # ======================
    # 対策③: 時間方向の間引き (Decimation)
    # ナイキスト周波数が高すぎるので、10ステップに1回だけ取り出す
    # ======================
    skip = 8
    Ez_global = Ez_global[::skip, :]
    dt = dt_original * skip  # 間引いた分、dtを大きくする！
    print(f"Reduced shape: {Ez_global.shape}")

    Nt, Nx = Ez_global.shape

    # ======================
    # 必須①：時間平均除去（ω=0対策）
    # ======================
    Ez_global = Ez_global - Ez_global.mean(axis=0, keepdims=True)

    window = np.hanning(Nt).astype(np.float32)[:, np.newaxis]
    Ez_global = Ez_global * window

    # ======================
    # FFT (一気に2次元FFTを実行)
    # ======================
    print("Running FFT...")
    # np.fft.fft2 を使うと2回のFFTを1行で書けます
    E_wk = np.fft.fft2(Ez_global)
    
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    omega = 2 * np.pi * np.fft.fftfreq(Nt, d=dt)

    # powerもfloat32に落としてメモリ節約
    power = (np.abs(E_wk)**2).astype(np.float32)
    
    # E_wkはもう使わないのでメモリ解放
    del E_wk
    gc.collect()

    # ======================
    # 必須②：ω>0 のみ
    # ======================
    om_mask = omega > 0
    k_mask = k > 0

    omega = omega[om_mask]
    k = k[k_mask]
    power = power[np.ix_(om_mask, k_mask)]

    # ======================
    # log 表示
    # ======================
    # ゼロ割りを防ぐために微小値を足す
    log_power = np.log10(power + 1e-20)

    # ======================
    # 軸の選択と描画
    # ======================
    use_normalized_axis = True

    if use_normalized_axis:
        k_plot = k * R_ce
        omega_plot = omega / omega_ce
        xlabel = r"$kR_{ce}$"
        ylabel = r"$\omega/\omega_{ce}$"
    else:
        k_plot = k
        omega_plot = omega
        xlabel = r"$k\,[\mathrm{rad/m}]$"
        ylabel = r"$\omega\,[\mathrm{rad/s}]$"

    print("Plotting...")
    plt.figure(figsize=(8,6))
    plt.pcolormesh(k_plot, omega_plot, log_power,
                   shading="auto", cmap="inferno")
    plt.colorbar(label=r"$\log_{10}$ power")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # 描画範囲を制限（高周波すぎるところは見ない）
    plt.xlim((0., 5.))
    plt.ylim((0., 10.))
    
    plt.tight_layout()
    plt.savefig("spectrum.png", dpi=300)
    plt.close()
    print("Done!")