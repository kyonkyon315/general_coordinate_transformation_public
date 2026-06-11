//#include "mpi.h"
#include <cmath>
//#include <random>

#include "../../supercomputer_instruments/axis_instantiator.h"

//#include "../include_super.h"
//#include "../supercomputer_instruments/axis_instantiator.h"
//int rank;

using Value = double;
using namespace std;

//計算空間の座標を設定します。
//Axis<ここには軸の通し番号をintで入力します。,ここには座標のローカルグリッド数をintで入力します,ここは軸の並列化数を書きます,ここはゴーストセル数>
//軸のグローバルなグリッド数はローカルグリッド数＊並列数です。
//全体をなめる計算においては、通し番号が小さいものほど、より外側のループを担当することになります。
//また、x,v空間で、∂x/∂v = 0である必要があります。（電流の計算を簡単に行うための措置です。）
//∂v/∂x = 0 は要求されていません。（例えば背景磁場に沿って速度空間の向きを変えたい時など）
//通し番号は重複することなく、互いに隣り合った0以上の整数である必要があります。また、0を含む必要があります。
//物理空間↔計算空間の写像は、全単射である必要があります。
#include "../../supercomputer_instruments/axis.h"
using Axis_z_ = Axis<0,2048/256,256,3>;
using Axis_vr = Axis<1, 128/  8,  8,3>;
using Axis_vt = Axis<2,  32    ,  1,3>;
using Axis_vp = Axis<3,  64    ,  1,3>;

//Δvz = ΔrΔt/2 =0.064*0.2/2=0.0064 : e-20

//担当するブロックの各軸の左端インデックス
//main関数内で設定される。グローバル変数で使うので、ここで定義している。

//電子分布関数の型を定義
//先頭に入力する型はテンソルの値の型です。その後に続く軸は、通し番号が小さいものほど左に入力してください。
#include "../../supercomputer_instruments/n_d_tensor_with_ghost_cell.h"
using DistributionFunction = NdTensorWithGhostCell<Value,Axis_z_,Axis_vr,Axis_vt,Axis_vp>;

//磁場の型を定義
#include "../../vec3.h"
using MagneticField = NdTensorWithGhostCell<Vec3<Value>,Axis_z_>;
//B(i,j).z=Bz(x=Δx i     ,t=Δt(j+1/2))
//B(i,j).x=Bx(x=Δx(i+1/2),t=Δt(j+1/2))
//B(i,j).y=By(x=Δx(i+1/2),t=Δt(j+1/2))

//電場の型を定義
using ElectricField = NdTensorWithGhostCell<Vec3<Value>,Axis_z_>;
//E(i,j).z=Ez(x=Δx(i+1/2),t=Δt j)
//E(i,j).x=Ex(x=Δx i     ,t=Δt j)
//E(i,j).y=Ey(x=Δx i     ,t=Δt j)

#include "../../pack.h"
using VeloPack = Pack<Axis_vr,Axis_vt,Axis_vp>;
//電流の型を定義
//電流は実空間のみのグリッドを持つので、Axis_vxは与えない。
//ただし、電流計算用の足し合わせで速度空間の情報が必要なので、Pack<Axis_vx>を与える。
#include "../../supercomputer_instruments/current.h"
using Current_type = Current<Vec3<Value>,VeloPack,Axis_z_>;
//current.at(i).x = j_x(x=Δx i)
//current.at(i).y = j_y(x=Δx i)
//current.at(i).z = j_z(x=Δx(i+1/2))

//電流計算が不要の時（磁場固定のときなど）はCurrentをNone_currentにしておく
//using Current = None_current;

/***********************************************
 * 物理空間と計算空間の関係を表す関数を書きます(始)*
 ***********************************************/
#include "../../normalization.h"
// --- グローバル定数とヘルパー関数の定義 ---
namespace Global{
    constexpr Value v_th_para = 5./8.;
    //constexpr Value v_th_para = 1.;
     
    constexpr Value grid_size_z_ = Norm::Param::debye_length/Norm::Base::x0;

    constexpr Value v_max = 6.* Norm::Param::v_thermal/Norm::Base::v0;
    constexpr Value grid_size_vr = v_max / (double)Axis_vr::num_global_grid;

    constexpr Value grid_size_vt =    M_PI / (double)(Axis_vt::num_global_grid);
    constexpr Value grid_size_vp = 2.*M_PI / (double)(Axis_vp::num_global_grid);

    constexpr Value phi_courant = grid_size_vr*grid_size_vt*grid_size_vp/4.;
    //dt = 0.01;
    //c = 333
    //c dt / 20=333/20*0.01=3.33/20=1.6/10=0.16
    //v = 16.5
    //v dt =16.5/100=0.16
}

//計算空間はグリッドサイズが１なので、それを意味のあるスケールに変換するクラスをつくります
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

    Value apply(const int calc_z_)const{ return Global::grid_size_z_ * (0.5 + (double)(z__start_id + calc_z_));}
};

class CalcVr_2_Vr{
private:
    const int vr_start_id;
    static int calc_start_id(const int my_world_rank){
        auto [axis_z_, axis_vr, axis_vt, axis_vp] = axis_instantiator<Axis_z_,Axis_vr,Axis_vt,Axis_vp>(my_world_rank);
        return axis_vr.L_id;
    }
public:
    CalcVr_2_Vr(const int my_world_rank):
        vr_start_id(calc_start_id(my_world_rank))
    {}

    Value apply(const int calc_vr)const{ return Global::grid_size_vr * (0.5 + (double)(vr_start_id + calc_vr));}
};

class CalcVt_2_Vt{
private:
    const int vt_start_id;
    static int calc_start_id(const int my_world_rank){
        auto [axis_z_, axis_vr, axis_vt, axis_vp] = axis_instantiator<Axis_z_,Axis_vr,Axis_vt,Axis_vp>(my_world_rank);
        return axis_vt.L_id;
    }
public:
    CalcVt_2_Vt(const int my_world_rank):
        vt_start_id(calc_start_id(my_world_rank))
    {}
    Value apply(const int calc_vt)const{ return Global::grid_size_vt * (0.5 + (double)(vt_start_id+calc_vt));}
};

class CalcVp_2_Vp{
private:
    const int vp_start_id;
    static int calc_start_id(const int my_world_rank){
        auto [axis_z_, axis_vr, axis_vt, axis_vp] = axis_instantiator<Axis_z_,Axis_vr,Axis_vt,Axis_vp>(my_world_rank);
        return axis_vp.L_id;
    }
public:
    CalcVp_2_Vp(const int my_world_rank):
        vp_start_id(calc_start_id(my_world_rank))
    {}
    Value apply(const int calc_vp)const{ return Global::grid_size_vp * (0.5 + (double)(vp_start_id+calc_vp));}
};



// --- 物理量クラス ---
//honestly_translateで計算座標↔物理座標の変換の式を定義します。
//それを用いてコンストラクタで各場所での値を事前計算してテーブルに格納します。（table.set_value(honestly_translate))
//シミュレーション中はテーブルを参照します。
//こちらも計算軸クラスと同様に通し番号を設定します。
using FullSliceGhost_r = Slice<-Axis_vr::L_ghost_length, Axis_vr::num_grid+Axis_vr::R_ghost_length>;
using FullSliceGhost_t = Slice<-Axis_vt::L_ghost_length, Axis_vt::num_grid+Axis_vt::R_ghost_length>;
using FullSliceGhost_p = Slice<-Axis_vp::L_ghost_length, Axis_vp::num_grid+Axis_vp::R_ghost_length>;
class Physic_z_
{
    const CalcZ__2_Z_ calc_z__2_z;
public:
    Physic_z_(const int my_world_rank):
        calc_z__2_z(my_world_rank)
    {}
    Value honestly_translate(const int calc_z,const int calc_vr,const int calc_vt)const{
        return calc_z__2_z.apply(calc_z);
    }
    Value translate(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return calc_z__2_z.apply(calc_z);
    }
    static const int label = 0;
};

class Physic_vx
{
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt,Axis_vp> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
    const CalcVp_2_Vp calc_vp_2_vp;

public:
    Value honestly_translate(const int calc_vr,const int calc_vt,const int calc_vp)const{
        // v_x = vr * cos(vt)
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        const Value vp = calc_vp_2_vp.apply(calc_vp);
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
    Value translate(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);    
    }
    static const int label = 1;
};

class Physic_vy
{
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt,Axis_vp> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
    const CalcVp_2_Vp calc_vp_2_vp;

public:
    Value honestly_translate(const int calc_vr,const int calc_vt,const int calc_vp)const{
        // v_x = vr * cos(vt)
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        const Value vp = calc_vp_2_vp.apply(calc_vp);
        return vr * sin(vt)*sin(vp);
    }

    Physic_vy(const int my_world_rank):
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
    Value translate(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);    
    }
    static const int label = 2;
};


class Physic_vz
{
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;

public:
    Value honestly_translate(const int calc_vr,const int calc_vt)const{
        // v_x = vr * cos(vt)
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        return vr * cos(vt);
    }

    Physic_vz(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            [this](const int calc_vr,const int calc_vt){return honestly_translate(calc_vr, calc_vt);}
        );
    }
    Value translate(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vr,calc_vt);    
    }
    static const int label = 3;
};


class Jacobi_Det{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;

    Value honestly_translate(const int calc_vr,const int calc_vt){
        // v_y = vr * sin(vt) 
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        return (vr*vr*sin(vt)) 
                *Global::grid_size_z_*Global::grid_size_vr*Global::grid_size_vt*Global::grid_size_vp;
    }
public:
    Jacobi_Det(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            [this](const int calc_vr,const int calc_vt){
                return honestly_translate(calc_vr, calc_vt);
            }
        );
    }

    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vr,calc_vt);    
    }
};

template<int... Ints>
class Integers{
public:
    static constexpr std::array data = {Ints...};

    static constexpr int at(const int id){
        return data[id];
    }

    template<int ID>
    static constexpr int at(){
        return data[ID];
    }
};

#include <vector>
#include <string>
#include <optional>
#include <array>
#include "../../utils/bin_saver.h"

// --- データの構造化 ---
struct Point2D {
    Value x, y;
};

struct CellPolygon {
    Point2D p0; // 左上 (x0, y0 に相当)
    Point2D p1; // 右上 (x1, y1 に相当)
    Point2D p2; // 左下 (x2, y2 に相当)
    Point2D p3; // 右下 (x3, y3 に相当)
    Value f;    // 物理量
};

// --- 固定軸（スライス）の設定用構造体 ---
struct SliceConfig {
    std::optional<int> z;
    std::optional<int> vr;
    std::optional<int> vt;
    std::optional<int> vp;
};

int main()
{
    const int n_proc = 2048;
    const int step = 40000;
    std::cout<<"visualize"<<std::endl;

    // ★ ここで固定する軸を直感的に設定できます！
    // 数値を入れればそのグリッドで固定、std::nullopt なら全領域をループします。
    SliceConfig slice = {
        .z  = 0,             // z軸は 0 で固定
        .vr = std::nullopt,  // vr軸はフリー（すべて舐める）
        .vt = std::nullopt,  // vt軸はフリー
        .vp = 0              // vp軸は 0 で固定
    };

    // バラバラの配列ではなく、構造体のベクターに集約
    std::vector<CellPolygon> polygons;
    
    // パフォーマンスのため、あらかじめメモリを確保しておくとなお良しです
    polygons.reserve(2*Axis_vr::num_global_grid * Axis_vt::num_global_grid);

    for(int world_rank = 0; world_rank < n_proc; ++world_rank){
        auto [axis_z_, axis_vr, axis_vt, axis_vp] = axis_instantiator<Axis_z_, Axis_vr, Axis_vt, Axis_vp>(world_rank);
        
        // --- 担当外ブロックのスキップ判定 ---
        if (slice.z  && (*slice.z  < axis_z_.L_id || axis_z_.R_id <= *slice.z))  continue;
        if (slice.vr && (*slice.vr < axis_vr.L_id || axis_vr.R_id <= *slice.vr)) continue;
        if (slice.vt && (*slice.vt < axis_vt.L_id || axis_vt.R_id <= *slice.vt)) continue;
        if (slice.vp && (*slice.vp < axis_vp.L_id || axis_vp.R_id <= *slice.vp)) continue;

        // --- ファイル読み込み ---
        const std::string filename = "/LARGE1/gr20001/b39211/Documents/general_coodinate_transformation/whistler_with_pertur/dist_func/"
                 "step" + std::to_string(step) + "_"
                 + std::to_string(axis_z_.block_id) + "_"
                 + std::to_string(axis_vr.block_id) + "_"
                 + std::to_string(axis_vt.block_id) + "_"
                 + std::to_string(axis_vp.block_id) + ".bin";

        NdTensorWithGhostCell<Value, Axis_z_, Axis_vr, Axis_vt, Axis_vp> data(world_rank);
        data.load_physical_fast(filename);
        
        Jacobi_Det jacobi_det(world_rank);
        const Physic_vx operator_x(world_rank);
        const Physic_vz operator_y(world_rank);

        // --- ループ範囲の設定 ---
        // 固定されている場合はそのローカルインデックスのみ、フリーの場合は0〜num_gridまで回す
        int z_start  = slice.z  ? (*slice.z  % Axis_z_::num_grid) : 0;
        int z_end    = slice.z  ? (z_start + 1)                   : Axis_z_::num_grid;
        
        int vp_start = slice.vp ? (*slice.vp % Axis_vp::num_grid) : 0;
        int vp_end   = slice.vp ? (vp_start + 1)                  : Axis_vp::num_grid;

        int vr_start = slice.vr ? (*slice.vr % Axis_vr::num_grid) : 0;
        int vr_end   = slice.vr ? (vr_start + 1)                  : Axis_vr::num_grid;

        int vt_start = slice.vt ? (*slice.vt % Axis_vt::num_grid) : 0;
        int vt_end   = slice.vt ? (vt_start + 1)                  : Axis_vt::num_grid;

        // ★ 頂点計算用ラムダ：指定した頂点の周囲4セルの中心座標の平均をとる
        auto get_vertex = [&](int vr, int vt, int z, int vp) -> Point2D {
            Point2D p = {0.0, 0.0};
            for (int d_vr : {-1, 0}) {
                for (int d_vt : {-1, 0}) {
                    p.x += operator_x.honestly_translate(vr + d_vr, vt + d_vt, vp);
                    p.y += operator_y.honestly_translate(vr + d_vr, vt + d_vt);
                }
            }
            p.x /= 4.0;
            p.y /= 4.0;
            return p;
        };

        // --- メインの計算ループ ---
        for (int z_local = z_start; z_local < z_end; ++z_local) {
            for (int vp_local = vp_start; vp_local < vp_end; ++vp_local) {
                for (int vr_local = vr_start; vr_local < vr_end; ++vr_local) {
                    for (int vt_local = vt_start; vt_local < vt_end; ++vt_local) {
                        
                        CellPolygon poly;
                        // 4隅の座標をスッキリ計算
                        poly.p0 = get_vertex(vr_local,     vt_local,     z_local, vp_local); 
                        poly.p1 = get_vertex(vr_local + 1, vt_local,     z_local, vp_local); 
                        poly.p2 = get_vertex(vr_local + 1, vt_local + 1, z_local, vp_local); 
                        poly.p3 = get_vertex(vr_local,     vt_local + 1, z_local, vp_local); 

                        poly.f = data.at(z_local, vr_local, vt_local, vp_local) / jacobi_det.at(z_local, vr_local, vt_local, vp_local);

                        polygons.push_back(poly);
                    }
                }
            }
        }
    }

    slice = {
        .z  = 0,             // z軸は 0 で固定
        .vr = std::nullopt,  // vr軸はフリー（すべて舐める）
        .vt = std::nullopt,  // vt軸はフリー
        .vp = Axis_vp::num_global_grid/2            // vp軸は pi で固定
    };

    for(int world_rank = 0; world_rank < n_proc; ++world_rank){
        auto [axis_z_, axis_vr, axis_vt, axis_vp] = axis_instantiator<Axis_z_, Axis_vr, Axis_vt, Axis_vp>(world_rank);
        
        // --- 担当外ブロックのスキップ判定 ---
        if (slice.z  && (*slice.z  < axis_z_.L_id || axis_z_.R_id <= *slice.z))  continue;
        if (slice.vr && (*slice.vr < axis_vr.L_id || axis_vr.R_id <= *slice.vr)) continue;
        if (slice.vt && (*slice.vt < axis_vt.L_id || axis_vt.R_id <= *slice.vt)) continue;
        if (slice.vp && (*slice.vp < axis_vp.L_id || axis_vp.R_id <= *slice.vp)) continue;

        // --- ファイル読み込み ---
        const std::string filename = "/LARGE1/gr20001/b39211/Documents/general_coodinate_transformation/whistler_with_pertur/dist_func/"
                 "step" + std::to_string(step) + "_"
                 + std::to_string(axis_z_.block_id) + "_"
                 + std::to_string(axis_vr.block_id) + "_"
                 + std::to_string(axis_vt.block_id) + "_"
                 + std::to_string(axis_vp.block_id) + ".bin";

        NdTensorWithGhostCell<Value, Axis_z_, Axis_vr, Axis_vt, Axis_vp> data(world_rank);
        data.load_physical_fast(filename);
        
        Jacobi_Det jacobi_det(world_rank);
        const Physic_vx operator_x(world_rank);
        const Physic_vz operator_y(world_rank);

        // --- ループ範囲の設定 ---
        // 固定されている場合はそのローカルインデックスのみ、フリーの場合は0〜num_gridまで回す
        int z_start  = slice.z  ? (*slice.z  % Axis_z_::num_grid) : 0;
        int z_end    = slice.z  ? (z_start + 1)                   : Axis_z_::num_grid;
        
        int vp_start = slice.vp ? (*slice.vp % Axis_vp::num_grid) : 0;
        int vp_end   = slice.vp ? (vp_start + 1)                  : Axis_vp::num_grid;

        int vr_start = slice.vr ? (*slice.vr % Axis_vr::num_grid) : 0;
        int vr_end   = slice.vr ? (vr_start + 1)                  : Axis_vr::num_grid;

        int vt_start = slice.vt ? (*slice.vt % Axis_vt::num_grid) : 0;
        int vt_end   = slice.vt ? (vt_start + 1)                  : Axis_vt::num_grid;

        // ★ 頂点計算用ラムダ：指定した頂点の周囲4セルの中心座標の平均をとる
        auto get_vertex = [&](int vr, int vt, int z, int vp) -> Point2D {
            Point2D p = {0.0, 0.0};
            for (int d_vr : {-1, 0}) {
                for (int d_vt : {-1, 0}) {
                    p.x += operator_x.honestly_translate(vr + d_vr, vt + d_vt, vp);
                    p.y += operator_y.honestly_translate(vr + d_vr, vt + d_vt);
                }
            }
            p.x /= 4.0;
            p.y /= 4.0;
            return p;
        };

        // --- メインの計算ループ ---
        for (int z_local = z_start; z_local < z_end; ++z_local) {
            for (int vp_local = vp_start; vp_local < vp_end; ++vp_local) {
                for (int vr_local = vr_start; vr_local < vr_end; ++vr_local) {
                    for (int vt_local = vt_start; vt_local < vt_end; ++vt_local) {
                        
                        CellPolygon poly;
                        // 4隅の座標をスッキリ計算
                        poly.p0 = get_vertex(vr_local,     vt_local,     z_local, vp_local); 
                        poly.p1 = get_vertex(vr_local + 1, vt_local,     z_local, vp_local); 
                        poly.p2 = get_vertex(vr_local + 1, vt_local + 1, z_local, vp_local); 
                        poly.p3 = get_vertex(vr_local,     vt_local + 1, z_local, vp_local); 

                        poly.f = data.at(z_local, vr_local, vt_local, vp_local) / jacobi_det.at(z_local, vr_local, vt_local, vp_local);

                        polygons.push_back(poly);
                    }
                }
            }
        }
    }


    // polygons のデータを BinSaver 等で書き出す処理をここに繋げます
    // ...
    BinSaver polygon_file;
    polygon_file.open("polygon_file.bin");

    size_t num_polygons = polygons.size();
    polygon_file.write(num_polygons);
    polygon_file.write_vec(polygons);

    return 0;
}
