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
    print(len(files))
    files = files[:128]

    # 各ブロック読み込み
    data_blocks = [load_np(f,x,y) for f in files]

    Ez_global = np.hstack(data_blocks)

    return Ez_global
def move_av(x,window):
    kernel = np.ones(window) / window

    y = np.convolve(x, kernel, mode='valid')
    return y
import numpy as np

def lowpass_time(E_tx, dt, omega_c):
    """
    E_tx : shape (Nt, Nx)
    dt   : time step
    omega_c : カット周波数（rad/s）
    """
    
    Nt = E_tx.shape[0]
    window = np.hanning(Nt)
    E_tx = E_tx * window[:, None]
    # FFT（時間方向）
    E_wx = np.fft.fft(E_tx, axis=0)

    # 周波数軸（rad/s）
    freq = np.fft.fftfreq(Nt, d=dt)   # Hz
    omega = 2 * np.pi * freq          # rad/s

    # マスク作成
    mask = np.abs(omega) <= omega_c

    # フィルタ適用
    E_wx_filtered = E_wx * mask[:, None]

    # 逆FFT
    E_tx_filtered = np.fft.ifft(E_wx_filtered, axis=0)

    # 実数化（元が実数なら）
    return E_tx_filtered.real

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
    beta = 8./6.
    a=np.sqrt(np.pi/2.)*(1-w)**2
    b=-beta*w/np.abs(k) +(1-beta**2)*beta*(w-1.)/np.abs(k)
    c= np.exp(-beta**2 *(w-1)**2/(k**2)/2.)
    return a*b*c

if __name__ == "__main__":

    # ファイル一覧取得a
    strage = "/LARGE1/gr20001/b39211/Documents/general_coodinate_transformation"
    Ex_global = load_merge(strage + "/whistler_with_pertur/Ex_t_blockid_z_*.bin",4900,8)
    Ey_global = load_merge(strage + "/whistler_with_pertur/Ey_t_blockid_z_*.bin",4900,8)
    Ez_global = load_merge(strage + "/whistler_with_pertur/Ez_t_blockid_z_*.bin",4900,8)

    plt.plot(Ez_global[:,0])
    plt.savefig("E(t,x=0).png")
    plt.show()

    #Ex_global = Ex_global[4000:,:]
    #Ey_global = Ey_global[4000:,:]
    # ======================
    # 物理パラメータ
    # ======================
    c = 3e8
    omega_pe = 564102.5
    omega_ce = omega_pe/4.
    v_th = 0.1 * c
    R_ce = v_th / omega_ce

    dx = 0.25 * R_ce
    dt = 0.1 / omega_ce

    #Ex_global=lowpass_time(Ex_global,dt,omega_ce*1.1)
    #Ey_global=lowpass_time(Ey_global,dt,omega_ce*1.1)

    Nt, Nx = Ex_global.shape

    time = np.arange(Nt) * 0.1

    # x方向のグリッド数
    Nx = Ex_global.shape[1]

    # フーリエ変換（x方向のみ）
    Ex_k = np.fft.fft(Ex_global, axis=1)
    Ey_k = np.fft.fft(Ey_global, axis=1)

    # k配列
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    E_abs = np.sqrt(np.abs(Ex_k)**2 + np.abs(Ey_k)**2)
    E_abs = np.abs(Ex_k)
    log_E_abs = np.log10(E_abs+1e-23)

    log_E_abs = np.stack([move_av(log_E_abs[:, j],601) for j in range(log_E_abs.shape[1])], axis=1)
    # 正のkだけ使う
    k_pos = k[:Nx//2]
    Ex_k_pos = Ex_k[:, :Nx//2]

    # 例：いくつかのkインデックス
    def find_k_index(k_array, k_target):
        return np.argmin(np.abs(k_array - k_target))

    k_targets = [0.23,0.31, 0.32, 0.39,0.405,0.47,0.55,0.63,0.71,0.79,0.87]  # 単位は 1/m など
    k_indices = [find_k_index(k, kt/R_ce) for kt in k_targets]

    Nt = log_E_abs.shape[0]
    time = np.arange(Nt) * dt*omega_ce

    plt.figure()

    i=0
    for idx in k_indices:
        plt.plot(time[1:], log_E_abs[1:, idx], label=f"k Rce ={k_targets[i]:.3e}")
        i=i+1

    plt.xlabel("t[/omega_ce]")
    plt.ylabel("log_10 |E(k,t)|")
    plt.legend()
    plt.title("Time evolution of E(k)")
    plt.savefig("growth_rate.png", dpi=300) 
    plt.show()

    #線形成長率を求める
    w = np.linspace(0.01, 1.0, 3000)
    kR = k_hat_R(w)

    gamma = analitic_growth_rate(kR,w)


    # ==========================================
    # 追加部分: 各波数における成長率の計算とプロット
    # ==========================================
 
    import scipy.stats as stats

    # ==========================================
    # 追加部分: 各波数における成長率と95%信頼区間の計算
    # ==========================================
    
    ln_E_abs = np.log(E_abs + 1e-23)
    ln_E_abs = np.stack([move_av(ln_E_abs[:, j],600) for j in range(ln_E_abs.shape[1])], axis=1)
    Nt, Nx = ln_E_abs.shape
    time = np.arange(Nt) * 0.1
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx/R_ce)
    
    growth_rates = np.zeros(Nx)
    ci_lower = np.zeros(Nx)  # 信頼区間の下限
    ci_upper = np.zeros(Nx)  # 信頼区間の上限
    
    # 自由度 (データ点数 - パラメータ数(傾きと切片の2つ))
    dof = len(time) - 2
    
    # 95%信頼区間用のt値 (両側検定なので 0.975 を指定)
    t_val = stats.t.ppf(0.995, dof)
    
    for i in range(Nx):
        # cov=True で共分散行列を取得
        p, cov = np.polyfit(time, ln_E_abs[:, i], 1, cov=True)
        slope = p[0]
        
        # 共分散行列の [0, 0] 成分が傾きの分散。その平方根が標準誤差。
        se_slope = np.sqrt(cov[0, 0])
        
        # 誤差のマージン ＝ t値 × 標準誤差
        margin_of_error = t_val * se_slope
        
        growth_rates[i] = slope
        ci_lower[i] = slope - margin_of_error
        ci_upper[i] = slope + margin_of_error
        
    # プロットが綺麗に繋がるように k の小さい順にソート
    sort_idx = np.argsort(k)
    k_sorted = k[sort_idx]
    growth_rates_sorted = growth_rates[sort_idx]
    ci_lower_sorted = ci_lower[sort_idx]
    ci_upper_sorted = ci_upper[sort_idx]
    print(ci_lower_sorted)
    print(ci_upper_sorted)
    
    # ==========================================
    # グラフのプロット
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # 95%信頼区間を薄い灰色で塗りつぶす
    plt.fill_between(k_sorted, ci_lower_sorted, ci_upper_sorted, color='lightgray', alpha=0.7, label='95% Confidence Interval')
    
    # 成長率のメインライン
    plt.plot(k_sorted, growth_rates_sorted, marker='.', linestyle='-', color='blue', label='Growth Rate $\gamma$', markersize=2)
    
    plt.xlabel('Wavenumber $k$ [rad/m]', fontsize=14)
    plt.ylabel('Growth Rate $\gamma$', fontsize=14)
    plt.title('Growth Rate vs Wavenumber with 95% Confidence Interval', fontsize=16)
    plt.grid(True)

    #線形解析解
    plt.plot(kR,gamma,color="red")
    
    # x=0, y=0 の基準線
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.xlim((0,2))
    plt.ylim((-0.01,0.01))
    plt.savefig("growth_rate_k_gamma.png", dpi=300) 
    plt.show()
