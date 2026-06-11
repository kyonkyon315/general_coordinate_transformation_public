import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
if __name__ == "__main__":
    # バイナリファイルの読み込み
    filename = "polygon_file.bin"

    with open(filename, "rb") as f:
        # 1. まずポリゴンの数 (C++の size_t は通常 8バイトの符号なし整数) を読み込む
        num_polygons = np.fromfile(f, dtype=np.uint64, count=1)[0]
        
        # 2. C++の構造体 `CellPolygon` に対応するデータ型を定義 (Value が double の場合)
        # 構造体: p0(x,y), p1(x,y), p2(x,y), p3(x,y), f
        dt = np.dtype([
            ('p0_x', np.float64), ('p0_y', np.float64),
            ('p1_x', np.float64), ('p1_y', np.float64),
            ('p2_x', np.float64), ('p2_y', np.float64),
            ('p3_x', np.float64), ('p3_y', np.float64),
            ('f',    np.float64)
        ])
        
        # 3. データを一括読み込み
        data = np.fromfile(f, dtype=dt, count=num_polygons)

    # 描画用のポリゴンリストと、色設定用の値リスト
    # 描画用のポリゴンリストと、色設定用の値リスト
    patches = []
    values = []

    for row in data:
        poly_vertices = np.array([
            [row['p0_x'], row['p0_y']], # p0
            [row['p1_x'], row['p1_y']], # p1
            [row['p2_x'], row['p2_y']], # p3 (右下)
            [row['p3_x'], row['p3_y']]  # p2 (左下)
        ])
        patches.append(Polygon(poly_vertices, closed=True))
        values.append(row['f'])

    # numpy配列に変換
    val_array = np.array(values)

    # ★ 変更点：最大値で割って 0.0 〜 1.0 に規格化する ★
    # （もし負の値が含まれる可能性がある場合は np.max(np.abs(val_array)) などで絶対値の最大をとります）
    #val_array = val_array+1e-10
    #val_array = np.log(val_array)
    #min_val = np.min(val_array)
    #val_array = val_array+min_val
    max_val = np.max(val_array)
    if max_val > 0:
        normalized_values = val_array / max_val
    else:
        normalized_values = val_array # 全て0などのゼロ除算回避

    # 描画設定
    fig, ax = plt.subplots(figsize=(10, 8))

    p = PatchCollection(patches, cmap='viridis', edgecolor='none') 

    # 規格化した値をセット
    p.set_array(normalized_values)

    # ★ 変更点：カラーマップの適用範囲を 0.0 〜 1.0 にカチッと固定する ★
    p.set_clim(0.0, 1.0)

    ax.add_collection(p)
    ax.autoscale() 

    cbar = fig.colorbar(p, ax=ax)
    # ラベルに最大値を記載しておくと、実際の物理スケールが分かって便利です
    cbar.set_label(f'Normalized f (Max: {max_val:.2e})')

    ax.set_aspect('equal')
    ax.set_xlabel('Vx')
    ax.set_ylabel('Vz')
    plt.title('Distribution Function (Normalized by Max),z=0,vy=0')
    plt.tight_layout()

    plt.savefig("dist_t=wara.png",dpi=300)