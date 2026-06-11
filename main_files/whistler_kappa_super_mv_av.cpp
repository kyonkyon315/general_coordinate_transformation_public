#include "mpi.h"
#include <cmath>
#include <random>

#include "../supercomputer_instruments/axis_instantiator.h"

//#include "../include_super.h"
//#include "../supercomputer_instruments/axis_instantiator.h"

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
#include "../supercomputer_instruments/axis.h"
using Axis_z_ = Axis<0, 512/ 32, 32,3>;
using Axis_vr = Axis<1, 256/ 16, 16,3>;
using Axis_vt = Axis<2,  64/  4,  4,3>;
using Axis_vp = Axis<3,  64/  1,  1,3>;
static constexpr int prr_num = Axis_z_::num_blocks * Axis_vr::num_blocks * Axis_vt::num_blocks * Axis_vp::num_blocks;
//Δvz = ΔrΔt/2 =0.064*0.2/2=0.0064 : e-20

//担当するブロックの各軸の左端インデックス
//main関数内で設定される。グローバル変数で使うので、ここで定義している。

//電子分布関数の型を定義
//先頭に入力する型はテンソルの値の型です。その後に続く軸は、通し番号が小さいものほど左に入力してください。
#include "../supercomputer_instruments/n_d_tensor_with_ghost_cell.h"
using DistributionFunction = NdTensorWithGhostCell<Value,Axis_z_,Axis_vr,Axis_vt,Axis_vp>;

//磁場の型を定義
#include "../vec3.h"
using MagneticField = NdTensorWithGhostCell<Vec3<Value>,Axis_z_>;
//B(i,j).z=Bz(x=Δx i     ,t=Δt(j+1/2))
//B(i,j).x=Bx(x=Δx(i+1/2),t=Δt(j+1/2))
//B(i,j).y=By(x=Δx(i+1/2),t=Δt(j+1/2))

//電場の型を定義
using ElectricField = NdTensorWithGhostCell<Vec3<Value>,Axis_z_>;
//E(i,j).z=Ez(x=Δx(i+1/2),t=Δt j)
//E(i,j).x=Ex(x=Δx i     ,t=Δt j)
//E(i,j).y=Ey(x=Δx i     ,t=Δt j)

#include "../pack.h"
using VeloPack = Pack<Axis_vr,Axis_vt,Axis_vp>;
//電流の型を定義
//電流は実空間のみのグリッドを持つので、Axis_vxは与えない。
//ただし、電流計算用の足し合わせで速度空間の情報が必要なので、Pack<Axis_vx>を与える。
#include "../supercomputer_instruments/current.h"
using Current_type = Current<Vec3<Value>,VeloPack,Axis_z_>;
//current.at(i).x = j_x(x=Δx i)
//current.at(i).y = j_y(x=Δx i)
//current.at(i).z = j_z(x=Δx(i+1/2))

//電流計算が不要の時（磁場固定のときなど）はCurrentをNone_currentにしておく
//using Current = None_current;

/***********************************************
 * 物理空間と計算空間の関係を表す関数を書きます(始)*
 ***********************************************/
#include "../normalization.h"
// --- グローバル定数とヘルパー関数の定義 ---
namespace Global{
    /*
    Paul et al "Nonlinear evolution of whistler waves excited by subtracted kappa distribution"
    https://doi.org/10.1063/5.0300815

    のシミュレーションを再現する
    */
    
    constexpr Value v_th_para = 1./ Norm::Param::alpha;
    // 論文のSubtracted Maxwellianパラメータ（必要に応じて変更）

    constexpr Value beta  = 0.479339;
    constexpr Value Delta = 0.5;
    constexpr Value kappa = 2.0;
    constexpr Value a = (Norm::Param::omega_ce/Norm::Param::omega_pe)*(Norm::Param::omega_ce/Norm::Param::omega_pe);

    constexpr Value grid_size_z_ = 0.25;

    constexpr Value v_max = 5.* Norm::Param::v_thermal/Norm::Base::v0;
    static_assert(v_max < Norm::Coef::c_tilde);

    constexpr Value grid_size_vr = v_max / (double)Axis_vr::num_global_grid;

    constexpr Value grid_size_vt =    M_PI / (double)(Axis_vt::num_global_grid);
    constexpr Value grid_size_vp = 2.*M_PI / (double)(Axis_vp::num_global_grid);

    constexpr Value phi_courant = grid_size_vr*grid_size_vt*grid_size_vp/4.;

    constexpr Value v_th_c = 1./ Norm::Coef::c_tilde;

    constexpr Value sigma_cold = 0.1;

    constexpr Value Nh_N0 = 1.;
    // parameters

    // thermal widths
    //dt = 0.02;
    //c = 10
    //c dt / 5 = 10/5*0.02 = 0.04
    //v = 6
    //v dt =0.12 |dz=0.25
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


/***********************************************
 * 計算軸を物理軸で微分した値の関数を書きます　(始)*
 ***********************************************/



class Z__diff_z_
{
public:
    Z__diff_z_(){}
    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return 1./Global::grid_size_z_;
    }
};

class Vr_diff_vx
{
private:
    NdTensorWithGhostCell<Value,Axis_vt,Axis_vp> table;
    const CalcVt_2_Vt calc_vt_2_vt;
    const CalcVp_2_Vp calc_vp_2_vp;
public:
    Value honestly_translate(const int calc_vt,const int calc_vp){
        // sinθ cosφ
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        const Value vp = calc_vp_2_vp.apply(calc_vp);
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


class Vr_diff_vy
{
private:
    NdTensorWithGhostCell<Value,Axis_vt,Axis_vp> table;
    const CalcVt_2_Vt calc_vt_2_vt;
    const CalcVp_2_Vp calc_vp_2_vp;
public:
    Value honestly_translate(const int calc_vt,const int calc_vp){
        // sinθ sinφ
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        const Value vp = calc_vp_2_vp.apply(calc_vp);
        return sin(vt) * sin(vp)/(double)Global::grid_size_vr;
    }
    Vr_diff_vy(const int my_world_rank):
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

class Vr_diff_vz
{
private:
    NdTensorWithGhostCell<Value,Axis_vt> table;
    const CalcVt_2_Vt calc_vt_2_vt;
public:
    Value honestly_translate(const int calc_vt){
        // v_y = vr * sin(vt) 
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        return std::cos(vt)/(double)Global::grid_size_vr;
    }
    Vr_diff_vz(const int my_world_rank):
        table(my_world_rank),
        calc_vt_2_vt(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_t>(
            [this](const int calc_vt){return honestly_translate(calc_vt);}
        );
    }

    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vt);
    }
};

class Vt_diff_vx
{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt,Axis_vp> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
    const CalcVp_2_Vp calc_vp_2_vp;

public:
    Value honestly_translate(const int calc_vr,const int calc_vt,const int calc_vp){
        // v_y = vr * sin(vt) 
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        const Value vp = calc_vp_2_vp.apply(calc_vp);
        return  cos(vt)*cos(vp)/(vr*(double)Global::grid_size_vt);
    }
    Vt_diff_vx(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank),
        calc_vp_2_vp(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t,FullSliceGhost_p>(
            [this](const int calc_vr,const int calc_vt,const int calc_vp){
                return honestly_translate(calc_vr, calc_vt, calc_vp);
            }
        );
    }

    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);
    }
};

class Vt_diff_vy
{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt,Axis_vp> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
    const CalcVp_2_Vp calc_vp_2_vp;

public:
    Value honestly_translate(const int calc_vr,const int calc_vt,const int calc_vp){
        // v_y = vr * sin(vt) 
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        const Value vp = calc_vp_2_vp.apply(calc_vp);
        return  cos(vt)*sin(vp)/(vr*(double)Global::grid_size_vt);
    }
    Vt_diff_vy(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank),
        calc_vp_2_vp(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t,FullSliceGhost_p>(
            [this](const int calc_vr,const int calc_vt,const int calc_vp){
                return honestly_translate(calc_vr, calc_vt, calc_vp);
            }
        );
    }

    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);
    }
};

class Vt_diff_vz
{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
public:
    Value honestly_translate(const int calc_vr,const int calc_vt){
        // v_y = vr * sin(vt) 
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        return - std::sin(vt)/(vr*(double)Global::grid_size_vt);
    }
    Vt_diff_vz(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            [this](const int calc_vr,const int calc_vt){return honestly_translate(calc_vr, calc_vt);}
        );
    }

    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vr,calc_vt);
    }
};

class Vp_diff_vx
{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt,Axis_vp> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
    const CalcVp_2_Vp calc_vp_2_vp;

public:
    Value honestly_translate(const int calc_vr,const int calc_vt,const int calc_vp){
        // v_y = vr * sin(vt) 
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        const Value vp = calc_vp_2_vp.apply(calc_vp);
        return  - sin(vp)/(vr*sin(vt)*(double)Global::grid_size_vp);
    }
    Vp_diff_vx(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank),
        calc_vp_2_vp(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t,FullSliceGhost_p>(
            [this](const int calc_vr,const int calc_vt,const int calc_vp){
                return honestly_translate(calc_vr, calc_vt, calc_vp);
            }
        );
    }

    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);
    }
};


class Vp_diff_vy
{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt,Axis_vp> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
    const CalcVp_2_Vp calc_vp_2_vp;

public:
    Value honestly_translate(const int calc_vr,const int calc_vt,const int calc_vp){
        // v_y = vr * sin(vt) 
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        const Value vp = calc_vp_2_vp.apply(calc_vp);
        return  cos(vp)/(vr*sin(vt)*(double)Global::grid_size_vp);
    }
    Vp_diff_vy(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank),
        calc_vp_2_vp(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t,FullSliceGhost_p>(
            [this](const int calc_vr,const int calc_vt,const int calc_vp){
                return honestly_translate(calc_vr, calc_vt, calc_vp);
            }
        );
    }

    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);
    }
};


#include "../independent.h"

/*******************************************************************
 * Jacobian行列を定義します。上で作成したクラスを行列風に並べてください。*
 * 互いに独立な軸の箇所（微分が０）はIndependent classを入れてください。*
 *                                                                 *
 * 具体的には、Jacobian[I,J]には「通し番号Iの計算軸」を「通し番号Jの物理*
 * 軸」で微分したものを入れてください。                               *
 *******************************************************************/
#include "../jacobian.h"

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
/**********************************************
 * 解くべき移流方程式を定義します。              *
 * df/dt + v_x df/dx + q/m(E+v*B)・∇_v f = 0 *
 * を例に定義の仕方を解説                       *
 **********************************************/

//移流項の定義
//------------------------------------------
// 1. v_x * df/dx
//------------------------------------------
class Fz_ {
private:
    Physic_vz physic_vz;
public:
    Fz_(const int my_world_rank):
        physic_vz(my_world_rank)
    {}
    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp) const {
        return physic_vz.translate(calc_z, calc_vr, calc_vt, calc_vp);
    }
};

//B(i,j).z=Bz(x=Δx i     ,t=Δt(j+1/2))
//B(i,j).x=Bx(x=Δx(i+1/2),t=Δt(j+1/2))
//B(i,j).y=By(x=Δx(i+1/2),t=Δt(j+1/2))

//E(i,j).z=Ez(x=Δx(i+1/2),t=Δt j)
//E(i,j).x=Ex(x=Δx i     ,t=Δt j)
//E(i,j).y=Ey(x=Δx i     ,t=Δt j)


//------------------------------------------
// 2. q/m (E + v×B)_x
//------------------------------------------

bool is_velo_left_edge(const int my_world_rank){
    auto [axis_z_, axis_vr, axis_vt, axis_vp] = axis_instantiator<Axis_z_,Axis_vr,Axis_vt,Axis_vp>(my_world_rank);
    return axis_vr.block_id == 0;
}

bool is_velo_right_edge(const int my_world_rank){
    auto [axis_z_, axis_vr, axis_vt, axis_vp] = axis_instantiator<Axis_z_,Axis_vr,Axis_vt,Axis_vp>(my_world_rank);
    return axis_vr.block_id == Axis_vr::num_blocks-1;
}

class Fvx {
private:
    const bool _is_velo_right_edge;
    const ElectricField& e_field;
    const MagneticField& m_field;
    const Physic_vz& physic_vz;
    const Physic_vy& physic_vy;

public:
    Fvx(const int my_world_rank,
        const ElectricField& e_field,
        const MagneticField& m_field,
        const Physic_vz& physic_vz,
        const Physic_vy& physic_vy
    ):
        _is_velo_right_edge(is_velo_right_edge(my_world_rank)),
        e_field(e_field),
        m_field(m_field),
        physic_vz(physic_vz),
        physic_vy(physic_vy)
    {}

    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp) const {
        //速度空間の境界でフラックスが0になるように、移流を反対称にする。
        if(_is_velo_right_edge && calc_vr == Axis_vr::num_grid){
            return - at(calc_z,  Axis_vr::num_grid-1,calc_vt, calc_vp);
        }
        else{
            const Value Ex = e_field.at(calc_z).x;
            const Value By = (  m_field.at(calc_z-1).y +
                                m_field.at(calc_z).y)/2.;//Yee格子
            const Value Bz = m_field.at(calc_z).z;

            const Value vz = physic_vz.translate(calc_z, calc_vr, calc_vt, calc_vp);
            const Value vy = physic_vy.translate(calc_z, calc_vr, calc_vt, calc_vp);

            return - (Ex + vy*Bz - vz*By);//電子の電荷が負なので - がつく
            //規格化したので移流項はExのみ
        }
    }
};


class Fvy {
private:
    const bool _is_velo_right_edge;
    const ElectricField& e_field;
    const MagneticField& m_field;
    const Physic_vz& physic_vz;
    const Physic_vx& physic_vx;

public:
    Fvy(const int my_world_rank,
        const ElectricField& e_field,
        const MagneticField& m_field,
        const Physic_vz& physic_vz,
        const Physic_vx& physic_vx
    ):
        _is_velo_right_edge(is_velo_right_edge(my_world_rank)),
        e_field(e_field),
        m_field(m_field),
        physic_vz(physic_vz),
        physic_vx(physic_vx)
    {}

    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp) const {
        //速度空間の境界でフラックスが0になるように、移流を反対称にする。
        if(_is_velo_right_edge && calc_vr == Axis_vr::num_grid){
            return - at(calc_z,  Axis_vr::num_grid-1,calc_vt, calc_vp);
        }
        else{
            //Yee格子を採用しているため、電場はｘ方向に、磁場はｔ方向に補間しなければならない。
            const Value Ey = e_field.at(calc_z).y;
            const Value Bx = (  m_field.at(calc_z-1).x +
                                m_field.at(calc_z).x)/2.;
            const Value Bz =    m_field.at(calc_z).z;

            const Value vz = physic_vz.translate(calc_z, calc_vr, calc_vt, calc_vp);
            const Value vx = physic_vx.translate(calc_z, calc_vr, calc_vt, calc_vp);

            return - (Ey + vz*Bx - vx*Bz);//電子の電荷が負なので - がつく
            //規格化したので移流項はExのみ
        }
    }
};

class Fvz {
private:
    const bool _is_velo_right_edge;
    const ElectricField& e_field;
    const MagneticField& m_field;
    const Physic_vx& physic_vx;
    const Physic_vy& physic_vy;
public:
    Fvz(const int my_world_rank,
        const ElectricField& e_field,
        const MagneticField& m_field,
        const Physic_vx& physic_vx,
        const Physic_vy& physic_vy
    ):
        _is_velo_right_edge(is_velo_right_edge(my_world_rank)),
        e_field(e_field),
        m_field(m_field),
        physic_vx(physic_vx),
        physic_vy(physic_vy)
    {}

    Value at(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp) const {
        //速度空間の境界でフラックスが0になるように、移流を反対称にする。
        if(_is_velo_right_edge && calc_vr == Axis_vr::num_grid){
            return - at(calc_z,  Axis_vr::num_grid-1,calc_vt,calc_vp);
        }
        else{
            //Yee格子を採用しているため、電場はｘ方向に、磁場はｔ方向に補間しなければならない。
            const Value Ez = (  e_field.at(calc_z-1).z +
                                e_field.at(calc_z).z)/2.;
            const Value Bx = (  m_field.at(calc_z-1).x +
                                m_field.at(calc_z).x)/2.;
            const Value By = (  m_field.at(calc_z-1).y +
                                m_field.at(calc_z).y)/2.;
            const Value vx = physic_vx.translate(calc_z, calc_vr, calc_vt, calc_vp);
            const Value vy = physic_vy.translate(calc_z, calc_vr, calc_vt, calc_vp);
            
            return - (Ez + vx*By - vy*Bx);//電子の電荷が負なので - がつく
            //規格化したので移流項はExのみ
        }
    }
};

/****************************************************************************
 * 次に、Flux計算機を選択します。今回は、Umeda2008を用います。
 ****************************************************************************/
#include "../schemes/umeda_2008_fifth_order.h"
#include "../schemes/umeda_2008.h"
//using Scheme = Umeda2008;
using Scheme = Umeda2008_5;
namespace Global{
    Scheme scheme;
}
/*****************************************************************************
 * 境界条件を設定します。
 * ここで、境界条件を設定することと、ゴーストセルの更新方法を設定することは同値です。
 * ユーザーの目的を満たすようなゴーストセルの更新方法を設定してください。
 * 
 * 例えば、x方向に周期境界条件を用いたいならば、ゴーストセルは次のように更新すべきだ
 * ということは自明でしょう。
 * f(-x) ← f(x_length-x)
 * f(x_length+x) ← f(x)
 * 
 * また、反射境界を設定する場合は次のようになります。
 * f(-x,v) ← f(x,-v)
 * f(x_length+x,v) ← f(x_length-x,-v)
 * 
 * Axes と同様、labelを付けます。
 * ここではグローバルインデックスで記述してください。
 * 
 ****************************************************************************/
//xは周期境界条件
class BoundaryCondition_z_
{
public:
    static const int label = 0;

    //吸収項などを実装するとき、ゴーストセルにほかのセルの値を代入するだけではなくなる。このときtrueにする。->その場合の動作は未定義
    static constexpr bool not_only_comm = false;

    //ghost_cell[calc_x,calc_vx] <- cell[calc_x_ + Axis_x_::num_grid,calc_vx] 
    template<int Index>
    static int left(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp){
        if constexpr(Index == 0){
            return calc_z + Axis_z_::num_global_grid;
        }
        else if constexpr(Index == 1){
            return calc_vr;
        }
        else if constexpr(Index == 2){
            return calc_vt;
        }
        else if constexpr(Index == 3){
            return calc_vp;
        }
        else return 0;
    }
    //ghost_cell[calc_x,calc_vx] <- cell[calc_x_ - Axis_x_::num_grid,calc_vx] 
    template<int Index>
    static int right(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp){
        if constexpr(Index == 0){
            return calc_z - Axis_z_::num_global_grid;
        }
        else if constexpr(Index == 1){
            return calc_vr;
        }
        else if constexpr(Index == 2){
            return calc_vt;
        }
        else if constexpr(Index == 3){
            return calc_vp;
        }
        else return 0;
    }
};
static_assert(Axis_vt::num_global_grid%2 == 0);
static_assert(Axis_vp::num_global_grid%2 == 0);

class BoundaryCondition_vr
{
public:
    static const int label = 1;
    static constexpr bool not_only_comm = false;

    template<int Index>
    static int left(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp){
        static_assert(Axis_vt::num_grid%2 == 0,"v_theta空間のグリッド数は偶数である必要がある");
        constexpr int vp_half_num_grid = Axis_vp::num_global_grid/2;
        if constexpr(Index == 0){
            return calc_z;
        }
        else if constexpr(Index == 1){
            return -calc_vr-1;
        }
        else if constexpr(Index == 2){
            return Axis_vt::num_global_grid - calc_vt - 1;
        }
        else if constexpr(Index == 3){
            const int index_vp=(
                calc_vp < vp_half_num_grid ? 
                calc_vp+vp_half_num_grid:
                calc_vp-vp_half_num_grid
            );
            return index_vp;
        }
        else return 0;
    }

    template<int Index>
    static int right(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp){
        if constexpr(Index == 0){
            return calc_z;
        }
        else if constexpr(Index == 1){
            return Axis_vr::num_global_grid - 1 - (calc_vr - Axis_vr::num_global_grid);
        }
        else if constexpr(Index == 2){
            return calc_vt;
        }
        else if constexpr(Index == 3){
            return calc_vp;
        }
        else return 0;
    }
};

class BoundaryCondition_vt
{
private:
    static constexpr int vp_half_num_grid = Axis_vp::num_global_grid/2;
public:
    static const int label = 2;
    static constexpr bool not_only_comm = false;

    template<int Index>
    static int left(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp){
        if constexpr(Index == 0){
            return calc_z;
        }
        else if constexpr(Index == 1){
            return calc_vr;
        }
        else if constexpr(Index == 2){
            return - calc_vt -1 ;
        }
        else if constexpr(Index == 3){
            const int index_vp=(
                calc_vp < vp_half_num_grid ? 
                calc_vp+vp_half_num_grid:
                calc_vp-vp_half_num_grid
            );
            return index_vp;
        }
        else return 0;
    }

    template<int Index>
    static int right(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp){
        if constexpr(Index == 0){
            return calc_z;
        }
        else if constexpr(Index == 1){
            return calc_vr;
        }
        else if constexpr(Index == 2){
            return 2*Axis_vt::num_global_grid - calc_vt -1;
        }
        else if constexpr(Index == 3){
            const int index_vp=(
                calc_vp < vp_half_num_grid ? 
                calc_vp+vp_half_num_grid:
                calc_vp-vp_half_num_grid
            );
            return index_vp;
        }
        else return 0;
    }
};

//phi は周期境界条件
class BoundaryCondition_vp
{
public:
    static const int label = 3;
    static constexpr bool not_only_comm = false;

    template<int Index>
    static int left(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp){
        if constexpr(Index == 0){
            return calc_z;
        }
        else if constexpr(Index == 1){
            return calc_vr;
        }
        else if constexpr(Index == 2){
            return calc_vt;
        }
        else if constexpr(Index == 3){
            return calc_vp + Axis_vp::num_global_grid;
        }
        else return 0;
    }

    template<int Index>
    static int right(const int calc_z,const int calc_vr,const int calc_vt,const int calc_vp){
        if constexpr(Index == 0){
            return calc_z;
        }
        else if constexpr(Index == 1){
            return calc_vr;
        }
        else if constexpr(Index == 2){
            return calc_vt;
        }
        else if constexpr(Index == 3){
            return calc_vp - Axis_vp::num_global_grid;
        }
        else return 0;
    }
};

/*--------------------------------------
 * Pack を用いて境界条件をまとめます。
 *----------------------------------------------*/
using BoundaryCondition = Pack<BoundaryCondition_z_, BoundaryCondition_vr, BoundaryCondition_vt, BoundaryCondition_vp>;

//電場、磁場についても境界条件をまとめる
class BoundaryCondition_M_z_
{
public:
    static const int label = 0;

    //吸収項などを実装するとき、ゴーストセルにほかのセルの値を代入するだけではなくなる。このときtrueにする。->その場合の動作は未定義
    static constexpr bool not_only_comm = false;

    //ghost_cell[calc_x,calc_vx] <- cell[calc_x_ + Axis_x_::num_grid,calc_vx] 
    template<int Index>
    static int left(const int calc_z){
        if constexpr(Index == 0){
            return calc_z + Axis_z_::num_global_grid;
        }
        else return 0;
    }
    //ghost_cell[calc_x,calc_vx] <- cell[calc_x_ - Axis_x_::num_grid,calc_vx] 
    template<int Index>
    static int right(const int calc_z){
        if constexpr(Index == 0){
            return calc_z - Axis_z_::num_global_grid;
        }
        else return 0;
    }
};

/*----------------------------------------------------------------------------
 * ターゲットとなる関数とboundary_conditionを用いてboundary_managerを作成します。
 *---------------------------------------------------------------------------*/



/****************************************************************************
 * 最後に、解くべき移流方程式を定義します。
 *
 *Advections、および発展させたい関数（ここではdist_func）を
 *用いてAdvectionEquationをインスタンス化します。これが、本シミュレーションにおける
 *ブラソフソルバーとして働きます。
 ****************************************************************************/

/*
fdtd 関連
*/


/*
fdtd関連の設定終わり。
*/
//subtracted Kappa distribution
/*

fdtd関連の設定終わり。

*/

// Subtracted-Kappa distribution
Value fM(const Value vx_tilde,const Value vy_tilde,const Value vz_tilde/*無次元量が入る*/){

    const Value v_perp2 = vx_tilde*vx_tilde + vy_tilde*vy_tilde;
    const Value v_para2 = vz_tilde*vz_tilde;
    constexpr Value kappa = Global::kappa;
    constexpr Value sigma_para = Global::v_th_para;
    constexpr Value sigma_perp = 1.;
    constexpr Value Delta = Global::Delta;
    constexpr Value beta = Global::beta;
    constexpr Value Nh_N0 = Global::Nh_N0;

    // main kappa component
    const Value kappa_term1 =
        std::pow(
            1.0
            + v_para2/(2. * kappa * sigma_para * sigma_para)
            + v_perp2/(2. * kappa * sigma_perp * sigma_perp),
            -(kappa + 1.0)
        );

    // subtracted component
    const Value kappa_term2 =
        std::pow(
            1.0
            + v_para2/(2. * kappa * sigma_para * sigma_para)
            + v_perp2/(2. * kappa * beta * sigma_perp * sigma_perp),
            -(kappa + 1.0)
        );

    // normalization constant
    // ∫f d^3v = Nh_N0 となるように規格化
    const Value norm =
        std::pow(2. * M_PI * kappa, 1.5)
        * sigma_para
        * sigma_perp
        * sigma_perp
        * std::tgamma(kappa - 0.5)
        / std::tgamma(kappa + 1.0);

    const Value hot_f = ((1-beta * Delta)/(1-beta) * kappa_term1 - (1-Delta)/(1-beta) * kappa_term2) / norm;
    
    constexpr Value sigma_cold = Global::sigma_cold;

    const Value cold_f = std::exp(
                -v_para2 /(2.*sigma_cold * sigma_cold) 
                    - v_perp2 /(2.* sigma_cold * sigma_cold)
            )/(sigma_cold * sigma_cold * sigma_cold * Utils::ConstExpr::sqrt(2.*M_PI)*(2.* M_PI));

    return Norm::Coef::Ne_tilde * (Nh_N0 * hot_f + (1.-Nh_N0) * cold_f);
}

void init_dist_and_poisson(const int my_world_rank,DistributionFunction& f,ElectricField& e_field){
    std::vector<Value> etas(Axis_z_::num_global_grid);

    std::mt19937 rng(12345 );
    std::uniform_real_distribution<Value> uni(-1.0,1.0);
    
    for(int i=0;i<Axis_z_::num_global_grid;++i){
        Value eta = uni(rng);  // z 依存ノイズ
        //Value eps = 1e-7;
        Value eps = 0.;
        Value base = 1.;
        etas[i]=(base + eps * eta);
    }

    auto [axis_z_, axis_vr, axis_vt, axis_vp] = axis_instantiator<Axis_z_,Axis_vr,Axis_vt,Axis_vp>(my_world_rank);
    const int L_id = axis_z_.L_id;

    Jacobi_Det jacobi_det(my_world_rank);
    CalcVr_2_Vr calc_vr_2_vr(my_world_rank);
    Physic_vx physic_vx(my_world_rank);
    Physic_vy physic_vy(my_world_rank);
    Physic_vz physic_vz(my_world_rank);

    for(int iz=0; iz<Axis_z_::num_grid; iz++){
        for(int ivr=0; ivr<Axis_vr::num_grid; ivr++){
            for(int ivt=0; ivt<Axis_vt::num_grid; ivt++){
                for(int ivp=0; ivp<Axis_vp::num_grid; ivp++){
                    Value vx = physic_vx.translate(iz, ivr, ivt, ivp);
                    Value vy = physic_vy.translate(iz, ivr, ivt, ivp);
                    Value vz = physic_vz.translate(iz, ivr, ivt, ivp);
                    f.at(iz,ivr,ivt,ivp)
                        = jacobi_det.at(iz,ivr,ivt,ivp)
                            *fM(vx,vy,vz) * etas[L_id + iz];
                    //やこびあんで計算空間にスケールする
                }
            }
        }
    }

    Value n=0.;
    for(int j=0;j<Axis_vr::num_global_grid;++j){
        for(int k=0;k<Axis_vt::num_global_grid;++k){
            for(int l=0;l<Axis_vp::num_global_grid;++l){
                Value vr = Global::grid_size_vr * (0.5 + (double)j);
                Value vt = Global::grid_size_vt * (0.5 + (double)k);
                Value vp = Global::grid_size_vp * (0.5 + (double)l);
                Value vx = vr * sin(vt) * cos(vp);
                Value vy = vr * sin(vt) * sin(vp);
                Value vz = vr * cos(vt) ;
                   
                    n += Global::grid_size_z_*Global::grid_size_vr*Global::grid_size_vt*Global::grid_size_vp
                        *vr*vr*sin(vt)
                    *fM(vx,vy,vz);
            }
        }
    }

    std::vector<Value> F(Axis_z_::num_global_grid);
    for(int i=0;i<Axis_z_::num_global_grid;++i){
        F[i]=n*etas[i];
    }

    std::vector<Value> E(Axis_z_::num_global_grid);
    Value n_e_tilde_avg = 0.0;
    for(int i=0;i<Axis_z_::num_global_grid;++i){
        n_e_tilde_avg += F[i];
    }
    n_e_tilde_avg /= (Value)F.size();
    if(my_world_rank==0){
        std::cout<<"n_e_tilde_avg = "<<n_e_tilde_avg;
        std::cout<<"n = "<<n<<"\n";
    }
    // 電場の積分
    E[0] = 0.0; // 基準値（ポテンシャル自由度）
    for(int i=0;i<Axis_z_::num_global_grid-1;i++){
        Value n_e_tilde=F[i];
        E[i+1]
            = E[i] 
            + Norm::Coef::poisson_coef * (n_e_tilde_avg - n_e_tilde)/* *gridsize(=1)*/;
    }
    
    // 平均を引いて、周期境界条件を調整
    double E_mean = 0.0;
    for(int i=0;i<Axis_z_::num_global_grid;i++) E_mean += E[i];
    E_mean /= (Value)(Axis_z_::num_global_grid);

    for(int i=0;i<Axis_z_::num_global_grid;i++) E[i] -= E_mean;
    for(int i=0;i<Axis_z_::num_grid;i++){
        //e_field.at(i).z = E[L_id + i];
        e_field.at(i).z = 0.;
    }
}

class SelfDiffuse_phi{
private:
    DistributionFunction& dist_func;
    static constexpr int mv_av_N = 2;
    static_assert(mv_av_N<=Axis_vp::R_ghost_length, "移動平均の量はゴーストセル長よりも短くないといけない。");

    std::vector<Value> buf;
    const bool is_left_edge; 

public:
    SelfDiffuse_phi(const int my_world_rank, DistributionFunction& dist_func): 
        dist_func(dist_func),
        is_left_edge(is_velo_left_edge(my_world_rank))
    {
        buf.resize(Axis_vp::num_grid);
    }

    void apply(){
        if(!is_left_edge)return;

        constexpr int W = 2 * mv_av_N + 1;
        const int P = Axis_vp::num_grid;

        for(int z=0; z<Axis_z_::num_grid; z++){

            const int r = 0;

            for(int t=0; t<Axis_vt::num_grid; t++){

                Value sum = 0.0;

                // 初期window
                for(int i=-mv_av_N; i<=mv_av_N; i++){
                    sum += dist_func.at(z,r,t,i);
                }

                buf[0] = sum;

                // sliding window
                for(int p=1; p<P; p++){

                    sum += dist_func.at(z,r,t,p + mv_av_N);
                    sum -= dist_func.at(z,r,t,p - mv_av_N - 1);

                    buf[p] = sum;
                }

                // 書き戻し
                for(int p=0; p<P; p++){
                    dist_func.at(z,r,t,p) = buf[p] / W;
                }
            }
        }
    }
};

class SelfDiffuse_theta{
private:

    DistributionFunction& dist_func;

    static constexpr int mv_av_N = 2;

    std::vector<Value> buf;
    const CalcVt_2_Vt calc_vt_2_vt;
    const bool is_left_edge; 
public:

    SelfDiffuse_theta(const int my_world_rank,DistributionFunction& dist_func): 
        dist_func(dist_func),
        calc_vt_2_vt(my_world_rank),
        is_left_edge(is_velo_left_edge(my_world_rank))
    {
        buf.resize(Axis_vt::num_grid);
    }

    void apply(){
        if(!is_left_edge)return;

        const int T = Axis_vt::num_grid;

        for(int z=0; z<Axis_z_::num_grid; z++){

            const int r = 0;

            for(int p=0; p<Axis_vp::num_grid; p++){

                Value before_sum = 0.0;

                for(int t=0; t<T; t++){
                    before_sum += dist_func.at(z,r,t,p);
                }

                // --------------------
                // weighted moving average
                // --------------------

                for(int t=0; t<T; t++){

                    Value num = 0.0;
                    Value den = 0.0;

                    for(int i=t-mv_av_N;
                            i<=t+mv_av_N;
                            i++){

                        const Value theta_i =
                            calc_vt_2_vt.apply(i);

                        const Value w =
                            std::sin(theta_i);

                        num += dist_func.at(z,r,i,p);

                        den += w;
                    }
                    const Value theta = calc_vt_2_vt.apply(t);

                    buf[t] = num * std::sin(theta) / den;
                }

                // --------------------
                // 総和補正
                // --------------------

                Value after_sum = 0.0;

                for(int t=0; t<T; t++){
                    after_sum += buf[t];
                }

                const Value corr =
                    before_sum / after_sum;

                // --------------------
                // 書き戻し
                // --------------------

                for(int t=0; t<T; t++){

                    dist_func.at(z,r,t,p)
                        = corr * buf[t];
                }
            }
        }
    }
};
//#include "../dispersion_relation.h"
void init_magnetic_field(const int my_world_rank, MagneticField& m_field){
    auto [axis_z_, axis_vr, axis_vt, axis_vp] = axis_instantiator<Axis_z_,Axis_vr,Axis_vt,Axis_vp>(my_world_rank);
    std::mt19937 rng(axis_z_.block_id);
    std::uniform_real_distribution<Value> uni(-1.0,1.0);
    Physic_z_ physic_z_(my_world_rank);

    for(int i=0;i<Axis_z_::num_grid;++i){
        const Value eps = 1e-12;
        m_field.at(i).x= Norm::Coef::B_tilde*(eps * uni(rng));
        m_field.at(i).y= Norm::Coef::B_tilde*(eps * uni(rng));
        
        //const Value z = physic_z_.translate(i, 0, 0, 0);
        //m_field.at(i).x= Norm::Coef::B_tilde*eps * std::sin(k*z);
        //m_field.at(i).y= Norm::Coef::B_tilde*eps * std::cos(k*z);
        
    }
}

void init_electric_field(const int my_world_rank, ElectricField& e_field){
    auto [axis_z_, axis_vr, axis_vt, axis_vp] = axis_instantiator<Axis_z_,Axis_vr,Axis_vt,Axis_vp>(my_world_rank);
    std::mt19937 rng(axis_z_.block_id);
    std::uniform_real_distribution<Value> uni(-1.0,1.0);
    
    
    //Physic_z_ physic_z_(my_world_rank);

    for(int i=0;i<Axis_z_::num_grid;++i){
        const Value eps = 1e-12;
        e_field.at(i).x= Norm::Coef::B_tilde*(eps * uni(rng));
        e_field.at(i).y= Norm::Coef::B_tilde*(eps * uni(rng));
        
        //const Value z = physic_z_.translate(i, 0, 0, 0);
        //e_field.at(i).x= omega/k * Norm::Coef::B_tilde*eps * std::cos(k*z);
        //e_field.at(i).y= - omega/k * Norm::Coef::B_tilde*eps * std::sin(k*z);
    }
}

#include "../supercomputer_instruments/advection_equation.h"
#include "../supercomputer_instruments/FDTD/fdtd_solver_1d.h"
#include "../supercomputer_instruments/boundary_manager.h"
#include "../utils/Timer.h"
#include "../utils/bin_saver.h"
#include "../calc_current_in_x_and_y.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    DistributionFunction dist_function(world_rank);
    ElectricField e_field(world_rank);
    MagneticField m_field(world_rank);
    Current_type current(world_rank);

    const Physic_z_ physic_z_(world_rank);
    const Physic_vx physic_vx(world_rank);
    const Physic_vy physic_vy(world_rank);
    const Physic_vz physic_vz(world_rank);

    using Operators = Pack<Physic_z_,Physic_vx,Physic_vy,Physic_vz>;

    Operators operators(physic_z_,physic_vx,physic_vy,physic_vz);

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

    Fz_ flux_z_(world_rank);
    Fvx flux_vx(world_rank,e_field,m_field,physic_vz,physic_vy);
    Fvy flux_vy(world_rank,e_field,m_field,physic_vz,physic_vx);
    Fvz flux_vz(world_rank,e_field,m_field,physic_vx,physic_vy);

    const BoundaryCondition_z_ boundary_condition_z_;
    const BoundaryCondition_vr boundary_condition_vr;
    const BoundaryCondition_vt boundary_condition_vt;
    const BoundaryCondition_vp boundary_condition_vp;

    const Pack boundary_condition(
        boundary_condition_z_,
        boundary_condition_vr,
        boundary_condition_vt,
        boundary_condition_vp
    );

    const BoundaryCondition_M_z_ boundary_condition_M_z_;

    const Pack boundary_condition_em(boundary_condition_M_z_);
    
    const Pack advections(flux_z_,flux_vx,flux_vy,flux_vz);
    AdvectionEquation equation(world_rank,dist_function,advections,jacobian,Global::scheme, current);
    
    FDTD_solver_1d fdtd_solver(e_field,m_field,current);
    CalcCurrent_1d
    current_calculator(
        current,
        dist_function,
        operators,
        Global::grid_size_z_);

    auto [axis_z_, axis_vr, axis_vt, axis_vp] = axis_instantiator<Axis_z_,Axis_vr,Axis_vt,Axis_vp>(world_rank);
    
    BoundaryManager boundary_manager(world_rank,world_size,dist_function,boundary_condition,axis_z_,axis_vr,axis_vt,axis_vp);
    BoundaryManager boundary_manager_e(world_rank,world_size,e_field,boundary_condition_em,axis_z_,axis_vr, axis_vt, axis_vp);
    BoundaryManager boundary_manager_m(world_rank,world_size,m_field,boundary_condition_em,axis_z_,axis_vr, axis_vt, axis_vp);
    
    //初期化

    //背景磁場の設定
    for(int i=0;i<Axis_z_::num_grid;i++){
        m_field.at(i).z = Norm::Coef::B_tilde;
    }

    init_dist_and_poisson(world_rank,dist_function,e_field);

    init_magnetic_field(world_rank, m_field);
    init_electric_field(world_rank, e_field);

    // 初期化後のゴーストセル更新（重要）
    boundary_manager.apply<Axis_z_>();
    boundary_manager.apply<Axis_vr>();
    boundary_manager.apply<Axis_vt>();
    boundary_manager.apply<Axis_vp>();
    boundary_manager_e.apply<Axis_z_>();
    boundary_manager_m.apply<Axis_z_>();

    const Value dt = 0.01;

    int num_steps = 100000;
    if(world_rank==0)std::cout<<"whistler_kappa_super.cpp\n";
    if(world_rank==0)std::cout<<"include cold electrons\n";
    if(world_rank==0)std::cout<<"numstep:"<<num_steps<<"\n";

    BinSaver ex_log;
    BinSaver ey_log;
    BinSaver ez_log;
    std::vector<double> Ex_buf(Axis_z_::num_grid);
    std::vector<double> Ey_buf(Axis_z_::num_grid);
    std::vector<double> Ez_buf(Axis_z_::num_grid);

    BinSaver bx_log;
    BinSaver by_log;
    BinSaver bz_log;
    std::vector<double> Bx_buf(Axis_z_::num_grid);
    std::vector<double> By_buf(Axis_z_::num_grid);
    std::vector<double> Bz_buf(Axis_z_::num_grid);
    if(axis_vr.block_id==0 && axis_vt.block_id==0 && axis_vp.block_id==0){
        std::string strage_path = "/LARGE1/gr20001/b39211/Documents/general_coodinate_transformation/whistler_with_pertur";
        ex_log.open(strage_path + "/Ex_t_blockid_z_" + std::to_string(axis_z_.block_id) + ".bin");
        ey_log.open(strage_path + "/Ey_t_blockid_z_" + std::to_string(axis_z_.block_id) + ".bin");
        ez_log.open(strage_path + "/Ez_t_blockid_z_" + std::to_string(axis_z_.block_id) + ".bin");
        ex_log.write(Axis_z_::num_grid*num_steps);
        ey_log.write(Axis_z_::num_grid*num_steps);
        ez_log.write(Axis_z_::num_grid*num_steps);

        bx_log.open(strage_path + "/Bx_t_blockid_z_" + std::to_string(axis_z_.block_id) + ".bin");
        by_log.open(strage_path + "/By_t_blockid_z_" + std::to_string(axis_z_.block_id) + ".bin");
        bz_log.open(strage_path + "/Bz_t_blockid_z_" + std::to_string(axis_z_.block_id) + ".bin");
        bx_log.write(Axis_z_::num_grid*num_steps);
        by_log.write(Axis_z_::num_grid*num_steps);
        bz_log.write(Axis_z_::num_grid*num_steps);
    }

    SelfDiffuse_phi self_diffuse_phi(world_rank, dist_function);
    SelfDiffuse_theta self_diffuse_theta(world_rank, dist_function);

    Timer timer;
    timer.start();
    constexpr int Maxwell_split = 5;
    int status;
    for(int i=0;i<num_steps;i++){
        if(world_rank==0 && i%100==0)std::cout<<i<<" current:"<<current.global_at(0).x<<","<<current.global_at(0).y<<","<<current.global_at(0).z<<std::endl;
        //if(world_rank==0)std::cout<<" current:"<<current.at(0).x<<","<<current.at(0).y<<","<<current.at(0).z<<"\n";
        //if(world_rank==0)std::cout<<"       B:"<<m_field.at(0).x<<","<<m_field.at(0).y<<","<<m_field.at(0).z<<"\n";
        //v(0), x(0), E(0), B(0), J(-1/2)
        status = 0;
        equation.solve<Axis_vr>(dt/2.);
        boundary_manager.apply<Axis_vr>();

        equation.solve<Axis_vt>(dt/2.);
        boundary_manager.apply<Axis_vt>();
        self_diffuse_theta.apply();
        boundary_manager.apply<Axis_vt>();

        equation.solve<Axis_vp>(dt/2.);
        boundary_manager.apply<Axis_vp>();
        self_diffuse_phi.apply();
        boundary_manager.apply<Axis_vp>();

        //v(1/2), x(0), E(0), B(0), J(-1/2)
        current.clear();

        equation.solve<Axis_z_>(dt);
        boundary_manager.apply<Axis_z_>();
        
        current_calculator.calc();
        current.compute_global_current();
        
        //v(1/2), x(1), E(0), B(0), J(1/2)

        fdtd_solver.develop_m(dt/(double)Maxwell_split/2., Global::grid_size_z_);
        boundary_manager_m.apply<Axis_z_>();
        for(int j=0;j<Maxwell_split-1;j++){
            fdtd_solver.develop_e(dt/(double)Maxwell_split, Global::grid_size_z_);
            boundary_manager_e.apply<Axis_z_>();
            fdtd_solver.develop_m(dt/(double)Maxwell_split, Global::grid_size_z_);
            boundary_manager_m.apply<Axis_z_>();
        }
        fdtd_solver.develop_e(dt/(double)Maxwell_split, Global::grid_size_z_);
        boundary_manager_e.apply<Axis_z_>();
        fdtd_solver.develop_m(dt/(double)Maxwell_split/2., Global::grid_size_z_);
        boundary_manager_m.apply<Axis_z_>();

        //v(1/2), x(1), E(1), B(1), J(1/2)
        equation.solve<Axis_vp>(dt/2.);
        boundary_manager.apply<Axis_vp>();
        self_diffuse_phi.apply();
        boundary_manager.apply<Axis_vp>();
        
        equation.solve<Axis_vt>(dt/2.);
        boundary_manager.apply<Axis_vt>();
        self_diffuse_theta.apply();
        boundary_manager.apply<Axis_vt>();
        
        equation.solve<Axis_vr>(dt/2.);
        boundary_manager.apply<Axis_vr>();
        //v(1), x(1), E(1), B(1), J(1/2)

        if(axis_vr.block_id==0 && axis_vt.block_id==0 && axis_vp.block_id==0 && i%10==0){
        //if(false){

            for(int ix=0; ix<Axis_z_::num_grid; ix++){
                Ex_buf[ix]=e_field.at(ix).x;
                Ey_buf[ix]=e_field.at(ix).y;
                Ez_buf[ix]=e_field.at(ix).z;
            }
            ex_log.write_vec(Ex_buf);
            ey_log.write_vec(Ey_buf);
            ez_log.write_vec(Ez_buf);
            if(i%1000==0){
                ex_log.flush();
                ey_log.flush();
                ez_log.flush();
            }

            for(int ix=0; ix<Axis_z_::num_grid; ix++){
                Bx_buf[ix]=m_field.at(ix).x;
                By_buf[ix]=m_field.at(ix).y;
                Bz_buf[ix]=m_field.at(ix).z;
            }
            bx_log.write_vec(Bx_buf);
            by_log.write_vec(By_buf);
            bz_log.write_vec(Bz_buf);
            if(i%1000==0){
                bx_log.flush();
                by_log.flush();
                bz_log.flush();
            }
        }
        if(i%1000 == 0){
            const std::string filename = "/LARGE1/gr20001/b39211/Documents/general_coodinate_transformation/whistler_with_pertur/dist_func/"
                 "step" + std::to_string(i)+"_"
                 + std::to_string(axis_z_.block_id) + "_"
                 + std::to_string(axis_vr.block_id) + "_"
                 + std::to_string(axis_vt.block_id) + "_"
                 + std::to_string(axis_vp.block_id)
                 + ".bin";
            dist_function.save_physical_fast(filename);
        }
    }
    timer.stop();
    if(world_rank==0)std::cout<<timer<<"\n";
    MPI_Finalize();
    return 0;
}
