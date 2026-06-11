import numpy as np
import matplotlib.pyplot as plt
import glob
import re
# block_id でソート
def extract_block_id(fname):
    return int(re.search(r'blockid_z_(\d+)', fname).group(1))

def load_np(filename,x,y):
    with open(filename, "rb") as f:
        _ = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float64)
        data = data[:x*y]
    return data.reshape(x, y)

def load_merge(filename,x,y):
    files = glob.glob(filename)
    files.sort(key=extract_block_id)

    # 各ブロック読み込み
    data_blocks = [load_np(f,x,y) for f in files]

    Ez_global = np.hstack(data_blocks)

    return Ez_global

# 無次元パラメータ
wp_wc = 4.0      # ωp / ωc
vth_c = 0.1      # v_th / c



def k_hat_R(w):
    denom = w * (w - 1.0)
    n2 = 1 - (wp_wc**2) / denom
    n2[n2 < 0] = np.nan
    return (w * vth_c) * np.sqrt(n2)

def k_hat_L(w):
    denom = w * (w + 1.0)
    n2 = 1 - (wp_wc**2) / denom
    n2[n2 < 0] = np.nan
    return (w * vth_c) * np.sqrt(n2)

def analitic_growth_rate(k,w):
    beta = 8./5.
    a=np.sqrt(np.pi/2.)*(1-w)**2
    b=-beta*w/np.abs(k) +(1-beta**2)*beta*(w-1.)/np.abs(k)
    c= np.exp(-beta**2 *(w-1)**2/(k**2)/2.)
    return a*b*c

if __name__ == "__main__":
    strage = "/LARGE1/gr20001/b39211/Documents/general_coodinate_transformation/"

    # ファイル一覧取得
    Ex_global = load_merge(strage+"whistler_with_pertur/Ex_t_blockid_z_*.bin",2700,8)
    Ey_global = load_merge(strage+"whistler_with_pertur/Ey_t_blockid_z_*.bin",2700,8)
    Ez_global = load_merge(strage+"whistler_with_pertur/Ez_t_blockid_z_*.bin",2700,8)

    R=2700
    L=0
    Ex_global = Ex_global[L:R,:]
    Ey_global = Ey_global[L:R,:]
    Ez_global = Ez_global[L:R,:]
    
    E_abs = np.sqrt(Ex_global**2 + Ey_global**2)


    #Ez_global=Ez_global[:600,:]
    
    # ======================
    # 物理パラメータ
    # ======================
    c = 3e8
    omega_pe = 5641.025
    omega_ce = omega_pe/4.
    v_th = 0.1 * c
    R_ce = v_th / omega_ce

    dx = 0.25 * R_ce
    dt = 0.1 / omega_ce

    Nt, Nx = Ez_global.shape

    time = np.arange(Nt) * 0.1

    # プロット
    plt.plot(time, E_abs[:,0])
    #plt.plot(Ex_global[1600,:])

    # 軸ラベル
    plt.xlabel("t[/omega_ce]")
    plt.ylabel("Ex[/ B_0 v_th]")
    plt.yscale("log")
    # 保存
    plt.savefig("wave_form.png", dpi=300)
    plt.close()

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

    # 無次元周波数 0~1 なのでLO,RX は入らない
    w = np.linspace(0.01, 1.0, 3000)
    kR = k_hat_R(w)
    #kL = k_hat_L(w)

    # プロット
    #plt.plot(kR, w, label="R-mode (whistler)",color = "green",lw = 1.)
    #plt.plot(kL, w, label="L-mode",color = "green",lw=1.)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim((0.,5.))
    plt.ylim((0.,10.))
    plt.tight_layout()
    plt.savefig("spectrum.png", dpi=300)  # ← これを追加
    plt.close()

    gamma = analitic_growth_rate(kR,w)

    plt.plot(kR,gamma)
    plt.xlim((-2,2))
    plt.savefig("linear_growth_rate.png",dpi=300)

    print(np.nanmin(kR), np.nanmax(kR))

