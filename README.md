# Generic-Vlasov-Solver
## Click [here](https://github.com/kyonkyon315/general_coordinate_transformation_public/edit/main/README_english.md) for the English version.

任意次元・任意座標系においてプラズマの運動論的（Kinetic）シミュレーションを行うための、汎用かつ高パフォーマンスなVlasovソルバーです。
> **【お知らせ】**
> 本リポジトリは、主要な開発を行っている非公開リポジトリから一般公開用に抽出・コピーしたものです。過去のコミット履歴や一部の内部向け情報等は含まれておりません。あらかじめご了承ください。

## 概要
本プロジェクトは、計算空間（座標軸、グリッド数、境界条件、並列数）をC++のテンプレート・メタ・プログラミングを用いて完全に抽象化し、最小限のコード変更で多種多様な次元・座標系（デカルト座標系、円筒座標系、ダイポール座標系などへの物理空間写像）のプラズマ波動シミュレーションを実行できるように設計されています。

解く方程式はVlasov 方程式とMaxwell 方程式のふたつです。Maxwell 方程式は一般的なFDTDで実装されています。

1D1V（1次元実空間・1次元速度空間）の基本的な静電波動から、スーパーコンピュータを用いたMPI並列環境での高次元・複雑座標系（例：円筒座標系におけるホイッスラー波）のシミュレーションまで幅広く対応しています。
> ### 目的・背景
> オーロラの発生メカニズム解明や、宇宙天気予報（宇宙飛行士の被ばく予防など）への応用を見据え、宇宙空間における電子の分布関数と電磁場の時間発展をシミュレーションするために開発しました。
> プラズマシミュレーションでは「PIC（Particle-in-cell）法」という、粒子の動きを一つずつ追う手法が一般的です。しかし、計算コストの制約から扱える粒子数に限界があり、結果に「統計的ノイズ」が乗ってしまう弱点がありました。そこで本プロジェクトでは、電子を個々の粒子ではなく「速度分布関数」という連続的な流体のように扱う「Vlasov（ブラソフ）法」を採用し、ノイズが完全にゼロとなる高精度なシミュレーションを実現しています
> ### 担当役割 
> フレームワークの根幹となるアーキテクチャ設計から、テンプレートメタプログラミングによる抽象化、MPI通信の自動化ロジック、および各物理シミュレーション（ラングミュア波、ホイッスラー波など）の検証コードの実装まで、**全工程を自身で担当**しました。
> ### 技術スタックとアーキテクチャ
> - **使用言語:** C++17
> - **並列計算API:** MPI (Message Passing Interface)
> - **アーキテクチャ設計・実装方法:**
>   - **テンプレートメタプログラミングによる抽象化:** 実行時のオーバーヘッドをなくすため、次元数やグリッドサイズ、境界条件をコンパイル時に確定させる設計を採用。
>   - **物理演算と計算空間の分離:** `Operators`（物理演算子）と `Advections`（物理移流項）を独立した `Pack` クラスでカプセル化。
>   - **MPI通信トポロジーの自動生成:** `Axis` テンプレートに入力された並列分割数と境界条件から、隣接ノードとの通信グラフ（ゴーストセルの交換ロジック）を自動構築するアーキテクチャ。これにより、ユーザー側はMPIを意識せずにシミュレーションを記述可能。
> ### 開発において直面した課題と解決方法
> **数値拡散による分布関数の崩れ**
> * **課題:**  従来のVlasovシミュレーション（直交座標系）では、磁場中を電子が高速で回転する「サイクロトロン運動」を計算する際、計算誤差の蓄積によって時間経過とともに分布関数の形状がぼやけて崩れてしまう（数値拡散）という致命的な問題がありました（下図参照）。
>   
>   <img width="661" height="255" alt="image" src="https://github.com/user-attachments/assets/abc42001-e7b7-4f3a-93b8-75afc9c6605b" />
> * **解決方法:**  サイクロトロン運動の本質が「円運動」であることに着目し、速度空間の計算に極座標系（半径と角度）を導入する独自のアプローチを考案・実装しました。これにより、複雑な回転運動が「角度方向への単なる平行移動」として計算されるため、半径方向の数値拡散を原理的に完全に防ぐことに成功しました。下図の通り、長時間計算を行っても分布関数の形状が維持されています。
>   
>   <img width="655" height="272" alt="image" src="https://github.com/user-attachments/assets/54df8120-f7ef-42a7-a724-66b3058d39c5" />
> 
## Vlasov方程式とは

Vlasov方程式は、プラズマ中の多数の荷電粒子（電子やイオン）の運動を、「個々の粒子」として追跡するのではなく、「分布関数」の統計的な変化として記述する方程式です。
### 1. 基本となる考え方：
位相空間における粒子の「密度」を定義します。これを分布関数 $f(\mathbf{x}, \mathbf{v}, t)$ と呼びます。 $f(\mathbf{x}, \mathbf{v}, t) d\mathbf{x}d\mathbf{v}$ は、時刻 $t$ において、位置 $\mathbf{x}$ 周りの微小体積 $d\mathbf{x}$ 、速度 $\mathbf{v}$ 周りの微小体積 $d\mathbf{v}$ に存在する粒子の数を表します。

### 2. 方程式の形
外力（電磁場）が存在する環境下での無衝突プラズマのVlasov方程式は、以下の形で表されます：

$$\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla_x f + \frac{q}{m} (\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \nabla_v f = 0$$

ここで、 $f$, $q$, $m$, $E$, $B$, $\mathbf{v}$, $\nabla_x$, $\nabla_v$ はそれぞれ、電子分布関数、素電荷、電子質量、電場、磁場、速度( $v_x, v_y, v_z$ )、実空間勾配($\frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \frac{\partial}{\partial z}$)、速度空間勾配($\frac{\partial}{\partial v_x}, \frac{\partial}{\partial v_y}, \frac{\partial}{\partial v_z}$)です。
この式の意味を紐解くと、以下の3つの項から成り立っています：

- 時間変化項 ($\frac{\partial f}{\partial t}$): ある観測点における分布関数の時間変化。
- 実空間移流項 ($\mathbf{v} \cdot \nabla_x f$): 粒子が速度 $\mathbf{v}$ を持っているために、位置 $\mathbf{x}$ が移動することによる変化。
- 速度空間移流項 ($\frac{q}{m} (\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \nabla_v f$): 電場 $\mathbf{E}$ や磁場 $\mathbf{B}$ から受けるローレンツ力によって、粒子の速度 $\mathbf{v}$ が変化することによる変化。

## 特徴
- **高い柔軟性と直感的な次元拡張**: `Axis` テンプレートクラスを定義するだけで、任意の次元数、グリッド数、境界条件、ゴーストセル数を設定可能。
- **型安全な高次元テンソル**: `NdTensorWithGhostCell` を用いて、多次元の分布関数や電磁場データを効率的かつ安全に管理。
- **物理演算と移流の分離**: 物理演算子（`Operators`）と物理移流項（`Advections`）を `Pack` クラスで独立してカプセル化し、計算スキームへの柔軟な組み込みを保証。
- **MPI並列計算のサポート**: 軸ごとに個別の並列分割数を指定可能（例：`Axis<ID, LocalGrids, MPI_Procs, Ghosts>`）。設定した境界条件を元に通信グラフを自動で生成するため、ユーザーはMPIのことは考える必要がない。
- **一般座標系・写像の対応**: 物理空間と計算空間が全単射で無図バレル変換であれば、基本的にどのような座標系でも設定可能。ユーザーは、軽量テンソルとヤコビアンを設定するだけでよい。

## 収録されているシミュレーション例
提供されているソースコードには、以下の検証・応用例が含まれています：

1. **`1D0V.cpp`**
   - 最もシンプルな1次元1軸の移流・境界条件テスト用コード。
   - 新たなVlasov scheme をテストするために使うこともできる。
<img width="400" height="200" alt="advection_simulation" src="https://github.com/user-attachments/assets/fddd6b4f-9ab8-4f26-bab5-a6a8d2a2e0f0" />

2. **`langmuir_wave.cpp`**
   - 1D1V 
   - デカルト座標系におけるラングミュア波の線形シミュレーション。
   - 下の図は、得られた電場 $E(x,t)$ を波数・周波数空間へフーリエ変換したものです。
<img width="300" height="360" alt="image (5)" src="https://github.com/user-attachments/assets/629669a2-3e3f-42b8-b8a5-c272efbf37d4" />

3. **`two_stream_instability.cpp`**
   - 1D1V 
   - プラズマにおける2流体不安定性（Two-stream instability）の成長と、位相空間（ $x-v_x$ ）における非線形渦（フェーズスペース・ホール）形成のシミュレーション。
<video 
  src="https://github.com/user-attachments/assets/a05559ea-120e-4537-91ee-18cb6a120e17"
  style="width: 20% !important; display: block !important;" 
  autoplay 
  loop 
  muted 
  playsinline
  preload="auto">
</video>

4. **`bernstein_mode_wave_super.cpp`**
   - 1D2V, 速度空間は極座標
   - 背景磁場に対して垂直に伝搬する静電場の時間発展を解く。
   - Bernstein mode wave の分散関係が得られる。
   - スーパーコンピューター向けのMPI並列化バージョン
   - 下の図は、得られた電場 $E(x,t)$ を波数・周波数空間へフーリエ変換したものです。バーンスタイン波の分散関係が確認できます。
<img width="320" height="240" alt="spectrum_bernstein_0 1c (1)" src="https://github.com/user-attachments/assets/506754b8-e764-44d6-b2e0-bd42d72c9300" />

5. **`whistler_cylinder_super.cpp`**
   - 1D3V, 速度空間は円筒座標
   - 不均一・異方性プラズマ中におけるホイッスラー波（Whistler wave）の伝搬や波動粒子相互作用をシミュレーション。
   - スーパーコンピューター向けのMPI並列化バージョン
   - Whistler-mode wave の分散関係が得られるほか、波の成長率が理論と一致することが確認できる。
<img width="320" height="240" alt="spectrum (6)" src="https://github.com/user-attachments/assets/7ae8a1ed-ea2f-4955-8413-d85077dac184" />
<img width="256" height="192" alt="growth_rate (2)" src="https://github.com/user-attachments/assets/41042eea-3e96-4c1f-97c4-3752a0c5e905" />
<img width="300" height="180" alt="growth_rate_k_gamma (4)" src="https://github.com/user-attachments/assets/9e821917-d5f9-4400-84cd-75b9d81a037d" />

## 必要環境・依存ライブラリ

* **C++17 以上** をサポートするコンパイラ（GCC, Clang, Intel Compiler など）
* **MPI ライブラリ**（MPI）

## 数学的枠組み

# 使いかた

本シミュレーションコードは、ユーザーが解きたい物理問題に合わせて、次元数や座標系をコンパイル時に柔軟に定義できる強力なフレームワークです。計算空間（一様なグリッド）と物理空間（実際の座標）を完全に分離し、ユーザーがその写像（変換ルール）を定義するだけで、複雑な座標系でのシミュレーションが可能になります。
`main/whistler_kappa_super.cpp`を例に、以下で使い方を説明します。
## 1. 次元数・グリッド数・並列数の設定
次元数は、計算に使用する「軸（Axis）」の定義とその組み合わせによって決定されます。コアコードを変更することなく、Axisを増減させるだけで1Dから多次元（例えば2X3Vなど）まで拡張可能です。軸の定義Axisテンプレートを使って、各計算軸を定義します。
```cpp
//Axis<通し番号, ローカルグリッド数, 並列化数, ゴーストセル数>
using Axis_z_ = Axis<0, 512/ 32, 32, 3>; // 空間 z軸
using Axis_vr = Axis<1, 256/ 16, 16, 3>; // 速度 r軸
using Axis_vt = Axis<2,  64/  4,  4, 3>; // 速度 theta軸
using Axis_vp = Axis<3,  64/  1,  1, 3>; // 速度 phi軸
```
通し番号: 0から始まる重複しない整数です。小さい番号ほどメモリの連続する外側のループを担当します。次元を追加する場合は、この番号を増やした新しいAxisを定義します。今回の場合ですと、合計の並列数が`32*16*4=2048`となります。また、ひとつのスレッドが担当するブロックのグリッドサイズは`[16,16,16,64]`となります。

## 2.物理量のコンテナの型定義
物理量の型`Value`および、1.で定義した軸`Axis_**`を`class NdTensorWithGhostCell`に渡すことで、対応した次元数・グリッド数・ゴーストセル数を持つテンソルの型を設定することができます。左に書いた軸ほど通し番号が小さい（外側のループになる）必要があります。
### 2.1. 速度分布関数f(z,vr,vt,vp)の型定義
```cpp
using Value = double;
using DistributionFunction = NdTensorWithGhostCell<Value, Axis_z_, Axis_vr, Axis_vt, Axis_vp>;
```
速度分布関数はスカラーですので、管理する値の型はdoubleとなります。また、速度分布関数は実空間・速度空間両方によって定義されるので、四つの軸すべてを用います。つまり、要素がスカラーの４次元テンソルとなります。

### 2.2. 電場E(z)、磁場B(z) の型定義
```cpp
#include "../vec3.h"
//磁場の型を定義
using MagneticField = NdTensorWithGhostCell<Vec3<Value>,Axis_z_>;
//B(i,j).z=Bz(x=Δx i     ,t=Δt(j+1/2))
//B(i,j).x=Bx(x=Δx(i+1/2),t=Δt(j+1/2))
//B(i,j).y=By(x=Δx(i+1/2),t=Δt(j+1/2))

//電場の型を定義
using ElectricField = NdTensorWithGhostCell<Vec3<Value>,Axis_z_>;
//E(i,j).z=Ez(x=Δx(i+1/2),t=Δt j)
//E(i,j).x=Ex(x=Δx i     ,t=Δt j)
//E(i,j).y=Ey(x=Δx i     ,t=Δt j)
```
電場・磁場の要素はベクトルであるため、先頭に`Vec3<Value>`を入れます。また、電場・磁場は実空間のみによって定義されるため、設定する軸はAxis_z_のみでよいです。つまり、要素がベクトルの１次元テンソルとなります。電場・磁場はYee格子を用いたFDTD法で時間発展を解くため、時間や位置の座標が半グリッドづつずれています。

## 3. 座標系の設定
本フレームワークは「計算空間」上で偏微分方程式を解きます。そのため、計算空間の座標 $(i, j, k, ...)$ から、実際の物理空間の座標 $(z, v_x, v_y, v_z)$ への 写像（全単射） と 軽量テンソルおよびヤコビアンをユーザー自身がクラスとして定義します。今回の例では、実空間が一般的な直線の座標、速度空間が極座標として設定されています。

### 3.1 計算空間 → 物理空間　の写像の実装

#### 3.1.1 計算空間 → 物理座標 $z$ への変換クラスの定義
```cpp
class CalcZ__2_Z_{
private:
    const int z__start_id;
    static int calc_start_id(const int my_world_rank){
        auto [axis_z_, axis_vr, axis_vt, axis_vp] = axis_instantiator<Axis_z_,Axis_vr,Axis_vt,Axis_vp>(my_world_rank);
        return axis_z_.L_id;
    }
public:
    CalcZ__2_Z_(const int my_world_rank):
        z__start_id(calc_start_id(my_world_rank))
    {}

    Value at(const int calc_z_)const{ return Global::grid_size_z_ * (0.5 + (double)(z__start_id + calc_z_));}
};

class Physic_z_
{
    const CalcZ__2_Z_ calc_z__2_z;
public:
    Physic_z_(const int my_world_rank):
        calc_z__2_z(my_world_rank)
    {}
    Value honestly_translate(const int calc_z,const int calc_vr,const int calc_vt)const{
        return calc_z__2_z.at(calc_z);
    }
    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return calc_z__2_z.at(calc_z);
    }
    static constexpr int label = 0;
};
```
z座標は等間隔に区切るので、グリッドサイズを掛ければよいです。ただし、入力されるcalc_z_はローカルid なので、スレッドが担当する範囲の左端のidを足す必要があります。一つ目の物理軸なので、`label = 0`とします。クラス`CalcZ__2_Z_`の実装は必須ではありません。

#### 3.1.2 計算空間 → 物理座標 $v_x$ の変換クラスの定義
```cpp
// 例：計算空間 (vr, vt, vp) から 物理空間 vx への写像 (vx = vr * sin(vt) * cos(vp))

class Physic_vx
{
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt,Axis_vp> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
    const CalcVp_2_Vp calc_vp_2_vp;

public:
    Value honestly_translate(const int calc_vr,const int calc_vt,const int calc_vp)const{
        // v_x = vr * cos(vt)
        const Value vr = calc_vr_2_vr.at(calc_vr);
        const Value vt = calc_vt_2_vt.at(calc_vt);
        const Value vp = calc_vp_2_vp.at(calc_vp);
        return vr * sin(vt)*cos(vp);
    }

    Physic_vx(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank),
        calc_vp_2_vp(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t,FullSliceGhost_p>(
            [this](const int calc_vr,const int calc_vt,const int calc_vp){
                return honestly_translate(calc_vr, calc_vt,calc_vp);
            }
        );
    }
    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);    
    }
    static const int label = 1;
};

```
`class CalcVt_2_Vt`、`class CalcVt_2_Vt`および`class CalcVp_2_Vp`の実装は必須ではありませんが、プログラミングの簡単化のために作っています。$v_x$ を求めるには $sin,cos$ などの重たい計算が入るので、事前計算したものを`table`に入れてLUTすると高速化が期待できます。２つ目の物理軸なので、`label = 1`とします。`Physic_vy`, `Physic_vz`についても同様に実装してください。`label` はそれぞれ`2`, `3`となります。

### 3.2 計算空間と物理空間の間の計量テンソルの実装

計量テンソルは以下のように書けます：
```math
\frac{\partial(z_c,v_r,v_\theta,v_\phi)}
     {\partial(z,v_x,v_y,v_z)}
=
\begin{pmatrix}
\frac{\partial z_c}{\partial z} & 0 & 0 & 0 \\
0 & \frac{\partial v_r}{\partial v_x} & \frac{\partial v_r}{\partial v_y} & \frac{\partial v_r}{\partial v_z} \\
0 & \frac{\partial v_\theta}{\partial v_x} & \frac{\partial v_\theta}{\partial v_y} & \frac{\partial v_\theta}{\partial v_z} \\
0 & \frac{\partial v_\phi}{\partial v_x} & \frac{\partial v_\phi}{\partial v_y} & 0
\end{pmatrix}
```
この行列の各要素を扱う`class`を実装していきます。
#### 3.2.1 計量テンソル要素の実装
以下に計量テンソル要素を扱うクラスを $\frac{\partial v_r}{\partial v_x}$ を例に示します。メンバ関数`.at()`で値を取得できる必要があります。`.at()`の引数は、計算空間のローカルグリッド４つです。この数は次元数に合わせて変更してください。
```cpp

class Vr_diff_vx
{
private:
    NdTensorWithGhostCell<Value,Axis_vt,Axis_vp> table;
    const CalcVt_2_Vt calc_vt_2_vt;
    const CalcVp_2_Vp calc_vp_2_vp;
public:
    Value honestly_translate(const int calc_vt,const int calc_vp){
        // sinθ cosφ
        const Value vt = calc_vt_2_vt.at(calc_vt);
        const Value vp = calc_vp_2_vp.at(calc_vp);
        return sin(vt) * cos(vp)/(double)Global::grid_size_vr;
    }
    Vr_diff_vx(const int my_world_rank):
        table(my_world_rank),
        calc_vt_2_vt(my_world_rank),
        calc_vp_2_vp(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_t,FullSliceGhost_p>(
            [this](const int calc_vt,const int calc_vp){
                return honestly_translate(calc_vt, calc_vp);
            }
        );
    }
    
    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vt,calc_vp);
    }
};
```
事前に計算しておき、LUT すると高速化が期待できます。他の要素についても同様に実装してください。

#### 3.2.2 計量テンソルのインスタンス化

`main()`関数内で以下のように計量テンソルをインスタンス化してください。

```cpp
const Independent independent;
const Z__diff_z_ z__diff_z_;
const Vr_diff_vx vr_diff_vx(world_rank);
const Vr_diff_vy vr_diff_vy(world_rank);
const Vr_diff_vz vr_diff_vz(world_rank);
const Vt_diff_vx vt_diff_vx(world_rank);
const Vt_diff_vy vt_diff_vy(world_rank);
const Vt_diff_vz vt_diff_vz(world_rank);
const Vp_diff_vx vp_diff_vx(world_rank);
const Vp_diff_vy vp_diff_vy(world_rank);

const Jacobian jacobian(
   z__diff_z_ , independent, independent, independent, 
   independent, vr_diff_vx , vr_diff_vy , vr_diff_vz ,
   independent, vt_diff_vx , vt_diff_vy , vt_diff_vz ,
   independent, vp_diff_vx , vp_diff_vy , independent
);
```
微分が０の箇所には`independent`を入力します。

## 4. 境界条件の設定
計算領域の境界におけるゴーストセルの更新方法（境界条件）を設定します。本フレームワークでは、インデックスの対応関係（どのセルの値をゴーストセルにコピーするか）を定義するだけで、MPI通信を含む境界処理が自動的に行われます。クラス内に left と right のテンプレート関数を定義し、グローバルインデックスベースでコピー元の座標を指定します。
### 4.1 周期境界条件の実装例（z軸）
もっともシンプルな周期境界条件の例です。領域の反対側のセルの値をゴーストセルに持ってきます。
```cpp
class BoundaryCondition_z_
{
public:
    static const int label = 0;

    // 吸収境界など、単なる値のコピー以外の特殊な処理を行う場合はtrueにしますが、通常はfalseです。
    static constexpr bool not_only_comm = false;

    // 左側のゴーストセルを更新するためのコピー元インデックスを返す
    template<int Index>
    static int left(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp){
        if constexpr(Index == 0){
            // z軸方向は右端（+ num_global_grid）のインデックスを参照
            return calc_z + Axis_z_::num_global_grid;
        }
        else if constexpr(Index == 1){ return calc_vr; }
        else if constexpr(Index == 2){ return calc_vt; }
        else if constexpr(Index == 3){ return calc_vp; }
        else return 0;
    }

    // 右側のゴーストセルを更新するためのコピー元インデックスを返す
    template<int Index>
    static int right(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp){
        if constexpr(Index == 0){
            // z軸方向は左端（- num_global_grid）のインデックスを参照
            return calc_z - Axis_z_::num_global_grid;
        }
        else if constexpr(Index == 1){ return calc_vr; }
        else if constexpr(Index == 2){ return calc_vt; }
        else if constexpr(Index == 3){ return calc_vp; }
        else return 0;
    }
};
```
### 4.2 極座標などの複雑な境界条件（参考）
`whistler_kappa_super.cpp` の速度空間は極座標 $(v_r, v_\theta, v_\phi)$ で定義されています。例えば $v_r < 0$ という「負の絶対値」の領域にアクセスしようとした場合、原点を通り越して反対側の角度 ($v_\theta \to \pi - v_\theta$, $v_\phi \to v_\phi + \pi$) のデータを参照する必要があります。本フレームワークでは、こうした複雑な空間トポロジーもインデックスの条件分岐で簡潔に記述できます（詳細はソースコードの BoundaryCondition_vr などを参照してください）。
### 4.3 境界条件のPack化
作成した各軸の境界条件クラスは、Pack を用いて一つにまとめます。
```cpp
using BoundaryCondition = Pack<BoundaryCondition_z_, BoundaryCondition_vr, BoundaryCondition_vt, BoundaryCondition_vp>;
```
## 5. 移流項（フラックス）の設定
Vlasov方程式の各項における移流項を定義します。式 $\frac{\partial f}{\partial t} + \nabla \cdot (\mathbf{F} f) = 0$ における $\mathbf{F}$ に該当する部分です。
### 5.1 ローレンツ力による速度空間移流の実装
例として、 $v_x$ 方向の移流（ $-\frac{e}{m}(\mathbf{E} + \mathbf{v} \times \mathbf{B})_x$ ）を実装します。Yee格子を用いているため、電場・磁場の参照点が半グリッドずれる点に注意して補間を行います。
```cpp
class Fvx {
private:
    const bool _is_velo_right_edge;
    const ElectricField& e_field;
    const MagneticField& m_field;
    const Physic_vz& physic_vz;
    const Physic_vy& physic_vy;

public:
    Fvx(const int my_world_rank, const ElectricField& e_field, const MagneticField& m_field,
        const Physic_vz& physic_vz, const Physic_vy& physic_vy) : /* ...初期化... */ {}

    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp) const {
        // 速度空間の境界で粒子が計算領域外に逃げないよう、フラックスを0にする処理
        if(_is_velo_right_edge && calc_vr == Axis_vr::num_grid){
            return - at(calc_z, Axis_vr::num_grid-1, calc_vt, calc_vp);
        }
        else{
            // Yee格子に基づき、必要な位置の電磁場を取得（補間）
            const Value Ex = e_field.at(calc_z).x;
            const Value By = (m_field.at(calc_z-1).y + m_field.at(calc_z).y)/2.;
            const Value Bz = m_field.at(calc_z).z;

            // 物理速度を取得
            const Value vz = physic_vz.at(calc_z, calc_vr, calc_vt, calc_vp);
            const Value vy = physic_vy.at(calc_z, calc_vr, calc_vt, calc_vp);

            // ローレンツ力 -(E + v×B)_x を計算（電子電荷が負のため - を付与）
            return - (Ex + vy*Bz - vz*By); 
        }
    }
};
```
定義したすべての移流項クラスを Pack でまとめます。
```cpp
const Pack advections(flux_z_, flux_vx, flux_vy, flux_vz);
```
## 6. 数値計算スキームと全体構築
フラックス計算に使用する高次精度補間スキームを選択し、ここまで定義したすべての要素（分布関数、移流項、ヤコビアン、スキーム）を `AdvectionEquation`（ソルバー本体）に渡してインスタンス化します。
```cpp
// 補間スキームの選択（例: Umeda 2008の3次精度スキーム）
#include "../schemes/umeda_2008.h"
using Scheme = Umeda2008;
namespace Global{ Scheme scheme; }

// AdvectionEquation (Vlasovソルバー) のインスタンス化
AdvectionEquation equation(world_rank, dist_function, advections, jacobian, Global::scheme, current);

// 境界条件管理マネージャーのインスタンス化
BoundaryManager boundary_manager(world_rank, world_size, dist_function, boundary_condition, axis_z_, axis_vr, axis_vt, axis_vp);
```
## 7. メインループと時間積分
メインループでは、Strang splitting（方向分離解法）を用いて時間発展させます。各軸ごとに独立して移流方程式を解くことで、多次元問題を1次元の組み合わせに帰着させます。Maxwell方程式を解くFDTDソルバーと連携し、以下のような手順で1ステップを進めます。`equation.solve<Axis_**>(dt)` を呼んだ直後には、必ず `boundary_manager.apply<Axis_**>()` を呼び出してゴーストセルをMPI通信で同期させます。
```cpp
for(int i=0; i<num_steps; i++){
    // 1. 速度空間の半ステップ進行 (dt/2)
    equation.solve<Axis_vr>(dt/2.); boundary_manager.apply<Axis_vr>();
    equation.solve<Axis_vt>(dt/2.); boundary_manager.apply<Axis_vt>();
    equation.solve<Axis_vp>(dt/2.); boundary_manager.apply<Axis_vp>();

    // 2. 電流のクリアと、実空間の1ステップ進行 (dt)
    current.clear();
    equation.solve<Axis_z_>(dt);    
    boundary_manager.apply<Axis_z_>();
    
    // 3. 電流の計算とMaxwell方程式 (FDTD) の進行
    current_calculator.calc();
    current.compute_global_current();
    // (FDTDソルバーによる電磁場の時間発展... 詳細はソース参照)

    // 4. 速度空間の残り半ステップ進行 (dt/2)
    equation.solve<Axis_vp>(dt/2.); boundary_manager.apply<Axis_vp>();
    equation.solve<Axis_vt>(dt/2.); boundary_manager.apply<Axis_vt>();
    equation.solve<Axis_vr>(dt/2.); boundary_manager.apply<Axis_vr>();

    // ※ 適宜、ログの書き出しやデータ保存を行う
}
```
これで、任意の座標系における汎用Vlasovシミュレーションを実行する準備が整いました。ユーザーの解きたい物理系に合わせて、グリッド解像度や初期条件（init_dist_and_poisson など）を調整して計算を実行してください。
