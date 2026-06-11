import numpy as np
import matplotlib.pyplot as plt
import glob
import re

if __name__ == "__main__":

    # ファイル一覧取得
    files = glob.glob("../output/bernstein/Ez_t_blockid_z_*.dat")

    # block_id でソート
    def extract_block_id(fname):
        return int(re.search(r'blockid_z_(\d+)', fname).group(1))

    files.sort(key=extract_block_id)

    # 各ブロック読み込み
    data_blocks = [np.loadtxt(f) for f in files]

    #data_blocks = [data_blocks[0]]
    # shape確認
    for i, d in enumerate(data_blocks):
        print(f"block {i}: {d.shape}")

    # 横方向に結合（z方向に並べる）
    Ez_global = np.hstack(data_blocks)
    print(Ez_global[8])
    print(Ez_global[9])
    print(Ez_global[10])
    print(Ez_global[11])
    print(Ez_global[12])
    # ======================
    # 物理パラメータ
    # ======================
    c = 3e8
    omega_pe = 564102.5
    omega_ce = omega_pe/4.
    v_th = 0.003 * c
    R_ce = v_th / omega_ce

    dx = 0.125 * R_ce
    dt = 20.* 0.0025 / omega_ce

    Nt, Nx = Ez_global.shape

    # ======================
    # 必須①：時間平均除去（ω=0対策）
    # ======================
    Ez_global = Ez_global - Ez_global.mean(axis=0, keepdims=True)

    window = np.hanning(Nt)[:, np.newaxis]
    Ez_global = Ez_global * window

    # ======================
    # FFT
    # ======================
    E_k = np.fft.fft(Ez_global, axis=1)
    k = 2*np.pi*np.fft.fftfreq(Nx, d=dx)

    E_wk = np.fft.fft(E_k, axis=0)
    omega = 2*np.pi*np.fft.fftfreq(Nt, d=dt)

    power = np.abs(E_wk)**2

    # ======================
    # 必須②：ω>0 のみ
    # ======================
    om_mask = omega > 0
    k_mask = k > 0

    omega = omega[om_mask]
    k = k[k_mask]
    power = power[np.ix_(om_mask, k_mask)]

    # ======================
    # kごと正規化
    # ======================
    #power /= power.max(axis=0, keepdims=True)

    # ======================
    # log 表示
    # ======================
    log_power = np.log10(power)

    # ======================
    # 軸の選択
    # ======================
    use_normalized_axis = True


    if use_normalized_axis:
        k_plot = k * R_ce
        omega_plot = omega / omega_ce
        xlabel = r"$kR_{ce}$"
        ylabel = r"$\omega/\omega_{ce}$"
        #ylim = (0, 2.0)
    else:
        k_plot = k
        omega_plot = omega
        xlabel = r"$k\,[\mathrm{rad/m}]$"
        ylabel = r"$\omega\,[\mathrm{rad/s}]$"
        ylim = (0, omega.max())

    # ======================
    # 描画
    # ======================
    plt.figure(figsize=(8,6))
    plt.pcolormesh(k_plot, omega_plot, log_power,
                shading="auto", cmap="inferno")
    plt.colorbar(label=r"$\log_{10}$ power")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim((0.,5.))
    plt.ylim((0.,10.))
    plt.tight_layout()
    plt.savefig("spectrum.png", dpi=300)  # ← これを追加
    plt.close()
