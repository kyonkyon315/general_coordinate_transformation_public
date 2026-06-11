#include <cmath>
#include <mpi.h>
#include <string>

using Value = double;
using namespace std;

//計算空間の座標を設定します。
//Axis<ここには軸の通し番号をintで入力します。,ここには座標のグリッドの数をintで入力します,3,3>
//全体をなめる計算においては、通し番号が小さいものほど、より外側のループを担当することになります。
//通し番号は重複することなく、互いに隣り合った0以上の整数である必要があります。また、0を含む必要があります。
//計算空間の軸なので、一律Δx=1であり、軸同士は直交しています。
//最後の3,3 >はゴーストセルのグリッド数です。
#include "../supercomputer_instruments/axis.h"
using Axis_vr = Axis<0,100/1,1,3>;
using Axis_vt = Axis<1,100/1,1,3>;

#include "../supercomputer_instruments/n_d_tensor_with_ghost_cell.h"
//電子分布関数の型を定義
//先頭に入力する型はテンソルの値の型です。その後に続く軸は、通し番号が小さいものほど左に入力してください。
using DistributionFunction = NdTensorWithGhostCell<Value,Axis_vr,Axis_vt>;

#include "../vec3.h"
//磁場の型を定義
using MagneticField = Vec3<Value>;
using ElectricField = Vec3<Value>;

#include "../none.h"
//電流計算が不要の時（磁場固定のときなど）はCurrentをNone_currentにしておく
using Current_type = None_current;

/***********************************************
 * 物理空間と計算空間の関係を表す関数を書きます(始)*
 ***********************************************/
#include "../normalization.h"

// --- グローバル定数の定義 ---
namespace Global{
    constexpr Value v_max = 10. * Norm::Param::v_thermal / Norm::Base::v0;
    constexpr Value grid_size_vr = v_max / Axis_vr::num_global_grid;

    constexpr Value grid_size_vt = 2.*M_PI / (double)(Axis_vt::num_global_grid);

    namespace Vr{
        static constexpr int M = 6;
        static constexpr Value C = grid_size_vr;
        static constexpr Value D = C * ((Value)M + 1.5);
        static constexpr Value A = 4. * C * C * ((Value)M + 1.0);
        static constexpr Value B = 2. * C * C * ((Value)M + 1.0);
        static constexpr Value delta = 2.;
    }
}
#include "../supercomputer_instruments/axis_instantiator.h"
//計算空間はグリッドサイズが１なので、それを意味のあるスケールに変換するクラスをつくります

using FullSliceGhost_r = Slice<-Axis_vr::L_ghost_length, Axis_vr::num_grid+Axis_vr::R_ghost_length>;
using FullSliceGhost_t = Slice<-Axis_vt::L_ghost_length, Axis_vt::num_grid+Axis_vt::R_ghost_length>;

static int calc_start_id_r(const int my_world_rank){
    auto [axis_vr, axis_vt] = axis_instantiator<Axis_vr,Axis_vt>(my_world_rank);
    return axis_vr.L_id;
}

class CalcVr_2_Vr {
private:
    NdTensorWithGhostCell<Value,Axis_vr> table;
    const int vr_start_id;

public:
    CalcVr_2_Vr(const int my_world_rank):
        table(my_world_rank),
        vr_start_id(calc_start_id_r(my_world_rank))
    {
        table.set_value_sliced<FullSliceGhost_r>(
            [this](double x){ return local_at(x); }
        );
    }

    //local_at の高速バージョン　ただし、整数しか受け取れない。
    Value at(const int calc_vr) const { return table.at(calc_vr); }

    Value local_at(const double calc_vr)const{
        return global_at(calc_vr + (double)vr_start_id);
    }
    
    static Value global_at(const double calc_vr){
        if(calc_vr < -0.5){
            return -global_at(-calc_vr - 1.);
        }
        const Value rb = (Value)Global::Vr::M + 0.5;

        // 内側と外側の関数
        const Value f_in = std::sqrt(Global::Vr::A * calc_vr + Global::Vr::B);
        const Value f_out = Global::Vr::C * calc_vr + Global::Vr::D;

        // ブレンド係数 S(r) の計算
        const Value S = 0.5 * (1.0 + std::tanh((calc_vr - rb) / Global::Vr::delta));

        // 滑らかな結合
        return (1.0 - S) * f_in + S * f_out;
    }
};

class CalcVt_2_Vt{
private:
    const int vt_start_id;
public:
    CalcVt_2_Vt(const int my_world_rank):
        vt_start_id(calc_start_id_r(my_world_rank))
    {}
    Value at(const int calc_vt)const{ return Global::grid_size_vt * (0.5 + (double)(vt_start_id+calc_vt));}
};


// --- 物理座標クラス ---
//honestly_translateで計算座標↔物理座標の変換の式を定義します。
//それを用いてコンストラクタで各場所での値を事前計算してテーブルに格納します。（table.set_value(honestly_translate))
//シミュレーション中はテーブルを参照します。
//こちらも計算軸クラスと同様に通し番号を設定します。

class Physic_xi1
{
public:
    Value honestly_translate(int calc_vr,int calc_vt)const{
        return (Value)calc_vr;
    }
    Physic_xi1(const int my_world_rank){}
    Value at(int calc_vr,int calc_vt)const{
        return (Value)calc_vr;
    }
};
class Physic_xi2
{
public:
    Value honestly_translate(int calc_vr,int calc_vt)const{
        return (Value)calc_vt;
    }
    Physic_xi2(const int my_world_rank){}
    Value at(int calc_vr,int calc_vt)const{
        return (Value)calc_vt;
    }
};

class Physic_vx
{
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt> table;
public:
    Physic_vx(const int my_world_rank):
        table(my_world_rank)
    {
        const CalcVr_2_Vr calc_vr_2_vr(my_world_rank);
        const CalcVt_2_Vt calc_vt_2_vt(my_world_rank);
        auto honestly_translate = [&](int calc_vr,int calc_vt){
            // v_x = vr * cos(vt)
            const Value vr = calc_vr_2_vr.at(calc_vr);
            const Value vt = calc_vt_2_vt.at(calc_vt);
            return vr * cos(vt);
        };
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            honestly_translate
        );
    }

    Value at(int calc_vr,int calc_vt)const{
        return table.at(calc_vr,calc_vt);    
    }
    static const int label = 0;
};

class Physic_vy
{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt> table;
    
public:
    Physic_vy(const int my_world_rank):
        table(my_world_rank)
    {
        const CalcVr_2_Vr calc_vr_2_vr(my_world_rank);
        const CalcVt_2_Vt calc_vt_2_vt(my_world_rank);
        auto honestly_translate = [&](int calc_vr,int calc_vt){
            // v_y = vr * sin(vt) 
            const Value vr = calc_vr_2_vr.at(calc_vr);
            const Value vt = calc_vt_2_vt.at(calc_vt);
            return vr * sin(vt);
        };
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            honestly_translate
        );
    }
    Value at(int calc_vr,int calc_vt)const{
        return table.at(calc_vr,calc_vt);    
    }
    static const int label = 1;
};

/***********************************************
 * 計算軸を物理軸で微分した値の関数を書きます　(始)*
 ***********************************************/
class Vr_diff_VR {
private:
    NdTensorWithGhostCell<Value,Axis_vr> table;
    const int vr_start_id;
public:
    Vr_diff_VR(const int my_world_rank):
        table(my_world_rank),
        vr_start_id(calc_start_id_r(my_world_rank))
    {
        //ここ、一般ユーザーには理解できない気もするが、やっていることは、iとi+1 の平均をとってi+0.5 での微分が正しくなるようにしている。
        
        using Axis_vr_buf_t = Axis<0,Axis_vr::num_global_grid,1,Axis_vr::L_ghost_length>; 
        NdTensorWithGhostCell<Value, Axis_vr_buf_t> buf(0);
        const int last_id = Axis_vr_buf_t::num_global_grid+Axis_vr_buf_t::R_ghost_length -1; 
        buf.at(last_id) = global_at((Value)last_id);
        for(int i=Axis_vr_buf_t::num_global_grid+Axis_vr_buf_t::R_ghost_length -2;
            i>=-Axis_vr_buf_t::L_ghost_length;
            i--)
        {
            buf.at(i) = 2.* global_at((Value)i + 0.5) - buf.at(i+1);
        }
        //buf.at(-1) = buf.at(0);
        table.set_value_sliced<FullSliceGhost_r>(
             [&](int x){ return buf.at(x + vr_start_id); }
        );
    }
    //local_at の高速バージョン　ただし、整数しか受け取れない。
    Value at(const int calc_vr) const { return table.at(calc_vr); }

    Value local_at(const double calc_vr)const{
        return global_at(calc_vr + (double)vr_start_id);
    }
    static Value global_at(const Value calc_vr){
        Value r_eff = calc_vr;
        if(calc_vr < -0.5){
            r_eff = (Value)(-calc_vr - 1);
            return global_at(r_eff);
        }

        const Value rb = (Value)Global::Vr::M + 0.5;
        const Value delta = Global::Vr::delta;

        // 各関数の評価
        const Value f_in = std::sqrt(Global::Vr::A * r_eff + Global::Vr::B);
        const Value f_out = Global::Vr::C * r_eff + Global::Vr::D;
        
        // 各関数の導関数 (df_in/dr, df_out/dr)
        const Value df_in = Global::Vr::A / (2.0 * f_in);
        const Value df_out = Global::Vr::C;

        // ブレンド係数とその導関数 (積の微分法則に必要)
        const Value tanh_val = std::tanh((r_eff - rb) / delta);
        const Value S = 0.5 * (1.0 + tanh_val);
        const Value dS_dr = 0.5 * (1.0 - tanh_val * tanh_val) / delta;

        // 全体の導関数 dR/dr の計算
        const Value dR_dr = (1.0 - S) * df_in + S * df_out + dS_dr * (f_out - f_in);

        // 求めたいのは dr/dR なので逆数を返す
        return 1.0 / dR_dr;
    }
};

class Vr_diff_vx
{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt> table;
public:
    Vr_diff_vx(const int my_world_rank):
        table(my_world_rank)
    {
        const CalcVt_2_Vt calc_vt_2_vt(my_world_rank);
        const Vr_diff_VR vr_diff_vR(my_world_rank);
        auto honestly_translate = [&](const int calc_vr,const int calc_vt){
            // v_y = vr * sin(vt) 
            const Value vt = calc_vt_2_vt.at(calc_vt);
            return std::cos(vt) * vr_diff_vR.at(calc_vr);
        };
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            honestly_translate
        );
    }
    Value at(int calc_vr,int calc_vt)const{
        return table.at(calc_vr,calc_vt);
    }
};

class Vr_diff_vy
{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt> table;
public:
    Vr_diff_vy(const int my_world_rank):
        table(my_world_rank)
    {
        const CalcVt_2_Vt calc_vt_2_vt(my_world_rank);
        const Vr_diff_VR vr_diff_vR(my_world_rank);
        auto honestly_translate = [&](const int calc_vr,const int calc_vt){
            // v_y = vr * sin(vt) 
            const Value vt = calc_vt_2_vt.at(calc_vt);
            return std::sin(vt) * vr_diff_vR.at(calc_vr);
        };
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            honestly_translate
        );
    }
    Value at(int calc_vr,int calc_vt)const{
        return table.at(calc_vr,calc_vt);
    }
};

class Vt_diff_vx
{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt> table;
    
public:
    Vt_diff_vx(const int my_world_rank):
        table(my_world_rank)
    {
        const CalcVr_2_Vr calc_vr_2_vr(my_world_rank);
        const CalcVt_2_Vt calc_vt_2_vt(my_world_rank);
        auto honestly_translate = [&](const int calc_vr,const int calc_vt){
            // v_y = vr * sin(vt) 
            const Value vr = calc_vr_2_vr.at(calc_vr);
            const Value vt = calc_vt_2_vt.at(calc_vt);
            return  - std::sin(vt)/(vr*(double)Global::grid_size_vt);
        };
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            honestly_translate
        );
    }

    Value at(int calc_vr,int calc_vt)const{
        return table.at(calc_vr,calc_vt);
    }
};

class Vt_diff_vy
{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt> table;
public:
    Vt_diff_vy(const int my_world_rank):
        table(my_world_rank)
    {
        const CalcVr_2_Vr calc_vr_2_vr(my_world_rank);
        const CalcVt_2_Vt calc_vt_2_vt(my_world_rank);
        auto honestly_translate = [&](const int calc_vr,const int calc_vt){
            // v_y = vr * sin(vt) 
            const Value vr = calc_vr_2_vr.at(calc_vr);
            const Value vt = calc_vt_2_vt.at(calc_vt);
            return  std::cos(vt)/(vr*(double)Global::grid_size_vt);
        };
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            honestly_translate
        );
    }
    Value at(int calc_vr,int calc_vt)const{
        return table.at(calc_vr,calc_vt);
    }
};

class Jacobi_Det{
private:
    NdTensorWithGhostCell<Value,Axis_vr> table;
    const int vr_start_id;
public:
    Jacobi_Det(const int my_world_rank):
        table(my_world_rank),
        vr_start_id(calc_start_id_r(my_world_rank))
    {
        table.set_value_sliced<FullSliceGhost_r>(
             [this](int x){ return local_at((Value)x,0.); }
        );
    }
    //local_at の高速バージョン　ただし、整数しか受け取れない。
    Value at(const int calc_vr,const int calc_vt) const { return table.at(calc_vr); }

    Value local_at(const double calc_vr,const double calc_vt)const{
        return global_at(calc_vr + (double)vr_start_id,calc_vt);
    }
    static Value global_at(const Value calc_vr,const Value calc_vt){
        const Value R_l = CalcVr_2_Vr::global_at(calc_vr - 0.5);
        const Value R_r = CalcVr_2_Vr::global_at(calc_vr + 0.5);
        return Global::grid_size_vt*(R_r*R_r - R_l*R_l) / 2.;
    }
};

class Jacobi_Det_xi{
public:
    Jacobi_Det_xi(const int my_world_rank){}

    Value at(const int calc_vr,const int calc_vt)const{
        return 1.;
    }
};

/*******************************************
 * 解くべき移流方程式を定義します。           *
 * df/dt + q/m(v*B)・∇_v f = 0 *
 * を例に定義の仕方を解説                    *
 *******************************************/
bool is_velo_left_edge(const int my_world_rank){
    auto [axis_vr, axis_vt] = axis_instantiator<Axis_vr,Axis_vt>(my_world_rank);
    return axis_vr.block_id == 0;
}

bool is_velo_right_edge(const int my_world_rank){
    auto [axis_vr, axis_vt] = axis_instantiator<Axis_vr,Axis_vt>(my_world_rank);
    return axis_vr.block_id == Axis_vr::num_blocks-1;
}
//移流項の定義
//------------------------------------------
// 1. q/m (E + v×B)_x
//------------------------------------------
class Fvx {
private:
    const bool _is_velo_left_edge;
    const bool _is_velo_right_edge;
    const MagneticField& m_field;
    const ElectricField& e_field;
    const Physic_vy& physic_vy;
public:
    Fvx(const int my_world_rank,
        const MagneticField& m_field,
        const ElectricField& e_field,
        const Physic_vy& physic_vy
    ):
        _is_velo_left_edge(is_velo_left_edge(my_world_rank)),
        _is_velo_right_edge(is_velo_right_edge(my_world_rank)),
        m_field(m_field),
        e_field(e_field),
        physic_vy(physic_vy)
    {}
    Value at(int calc_vr, int calc_vt) const {
        //if(_is_velo_left_edge && calc_vr==-1){
        //    return - at(0,calc_vt);
        //}
        const Value vy = physic_vy.at(calc_vr, calc_vt);
        return - (vy*m_field.z + e_field.x);//電子の電荷が負なので - がつく
    }
};

//------------------------------------------
// 2. q/m (E + v×B)_x
//------------------------------------------
class Fvy {
private:
    const bool _is_velo_left_edge;
    const bool _is_velo_right_edge;
    const MagneticField& m_field;
    const ElectricField& e_field;
    const Physic_vx& physic_vx;
public:
    Fvy(const int my_world_rank,
        const MagneticField& m_field,
        const ElectricField& e_field,
        const Physic_vx& physic_vx
    ):
        _is_velo_left_edge(is_velo_left_edge(my_world_rank)),
        _is_velo_right_edge(is_velo_right_edge(my_world_rank)),
        m_field(m_field),
        e_field(e_field),
        physic_vx(physic_vx)
    {}
    Value at(int calc_vr, int calc_vt) const {
        //if(_is_velo_left_edge && calc_vr==-1){
        //    return - at(0,calc_vt);
        //}
        const Value vx = physic_vx.at(calc_vr, calc_vt);
        return vx*m_field.z - e_field.y;
    }
};
/****************************************************************************
 * 次に、Flux計算機を選択します。今回は、Umeda2008を用います。
 ****************************************************************************/
//#include "../schemes/umeda_2008_fifth_order.h"
#include "../schemes/umeda_2008.h"
//using Scheme = Umeda2008_5;
using Scheme = Umeda2008;
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
 * 
 ****************************************************************************/

class BoundaryCondition_vr
{
public:
    static const int label = 0;
    static constexpr bool not_only_comm = false;
    
    template<int Index>
    static int left(const int calc_vr,const int calc_vt){
        static_assert(Axis_vt::num_grid%2 == 0,"v_theta空間のグリッド数は偶数である必要がある");
        constexpr int vt_half_num_grid = Axis_vt::num_global_grid/2;
        
        if constexpr(Index == 0){
            return -calc_vr-1;
        }
        else if constexpr(Index == 1){
            const int index_vt=(
                calc_vt < vt_half_num_grid ? 
                calc_vt+vt_half_num_grid:
                calc_vt-vt_half_num_grid
            );
            return index_vt;
        }
        else return -1000;
    }

    template<int Index>
    static int right(const int calc_vr,const int calc_vt){
        if constexpr(Index == 0){
            return Axis_vr::num_global_grid - 1 - (calc_vr - Axis_vr::num_global_grid);
        }
        else if constexpr(Index == 1){
            return calc_vt;
        }
        else return -1000;
    }
};

//thetaは周期境界条件
class BoundaryCondition_vt
{
public:
    static const int label = 1;
    
    static constexpr bool not_only_comm = false;

    template<int Index>
    static int left(const int calc_vr,const int calc_vt){
        if constexpr(Index == 0){
            return calc_vr;
        }
        else if constexpr(Index == 1){
            return calc_vt + Axis_vt::num_global_grid;
        }
        else return 0;
    }

    template<int Index>
    static int right(const int calc_vr,const int calc_vt){
        if constexpr(Index == 0){
            return calc_vr;
        }
        else if constexpr(Index == 1){
            return calc_vt - Axis_vt::num_global_grid;
        }
        else return 0;
    }
};
#include "../pack.h"
/*--------------------------------------
 * Pack を用いて境界条件をまとめます。
 *----------------------------------------------*/
using BoundaryCondition = Pack<BoundaryCondition_vr, BoundaryCondition_vt>;
/*
Value fM(const Value vx_tilde,const Value vy_tilde){
    const Value U = Global::v_max/10.;
    return Norm::Coef::Ne_tilde * std::exp(-((vx_tilde-U)*(vx_tilde-U)+vy_tilde * vy_tilde) /2.)
    //return std::exp(-vr_tilde * vr_tilde /2.)
           /(2.* M_PI );
    //Ne_tilde = int f_tilde dv_tilde^3
}*/

Value fM(const Value vx_tilde,const Value vy_tilde){
    if( -Global::v_max/10. <vx_tilde && vx_tilde < Global::v_max/2. && 
        -Global::v_max/5. <vy_tilde && vy_tilde < Global::v_max/5. ){
        return 1.;
    }
    return 0.;
}

void init(int my_world_rank,const Jacobi_Det& jacobi_det,DistributionFunction& dist_function){
    Physic_vx physic_vx(my_world_rank);
    Physic_vy physic_vy(my_world_rank);
    for(int i=0;i<Axis_vr::num_grid;i++){
        for(int j=0;j<Axis_vt::num_grid;j++){
            const Value vx = physic_vx.at(i, j);
            const Value vy = physic_vy.at(i, j);
            dist_function.at(i,j) = fM(vx,vy)*jacobi_det.at(i, j);
        }
    }
}

#include "../supercomputer_instruments/advection_equation.h"
#include "../supercomputer_instruments/FDTD/fdtd_solver_1d.h"
#include "../supercomputer_instruments/boundary_manager.h"
#include "../jacobian.h"

#include "../projected_saver_2D.hpp"
#include "../utils/Timer.h"
//#include "../utils/polar_utils.h"
#include "../singularity_processor/singularity_processor.h"

void print(PolarCoordinates::SingularityProcessor & lut,int m,int n){
    std::cout << "Dynamic S_{"<<m<<","<< n<<"} = " << std::setprecision(16) << lut.calc_area(m,n) << std::endl; 
    //return lut.calc_area(E_dynamic * dt,0.,m,n);
}
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    DistributionFunction dist_function(world_rank);
    MagneticField m_field(world_rank);
    ElectricField e_field(world_rank);
    Current_type current;

    m_field.z = 0.;
    e_field.x = 1.3;
    e_field.y = 0.5;

    const Physic_vx physic_vx(world_rank);
    const Physic_vy physic_vy(world_rank);

    using Operators = Pack<Physic_vx,Physic_vy>;

    Operators operators(physic_vx,physic_vy);

    const Independent independent;
    const Vr_diff_vx vr_diff_vx(world_rank);
    const Vr_diff_vy vr_diff_vy(world_rank);
    const Vt_diff_vx vt_diff_vx(world_rank);
    const Vt_diff_vy vt_diff_vy(world_rank);

    const Jacobian jacobian(
        vr_diff_vx , vr_diff_vy, 
        vt_diff_vx , vt_diff_vy 
    );

    

    Fvx flux_vx(world_rank,m_field,e_field,physic_vy);
    Fvy flux_vy(world_rank,m_field,e_field,physic_vx);

    const BoundaryCondition_vr boundary_condition_vr;
    const BoundaryCondition_vt boundary_condition_vt;

    const Pack boundary_condition(
        boundary_condition_vr,
        boundary_condition_vt
    );

    
    const Pack advections(flux_vx,flux_vy);
    AdvectionEquation equation(world_rank,dist_function,advections,jacobian,Global::scheme, current);
    
    auto [axis_vr, axis_vt] = axis_instantiator<Axis_vr,Axis_vt>(world_rank);
    
    BoundaryManager boundary_manager(world_rank,world_size,dist_function,boundary_condition,axis_vr,axis_vt);
 

    Jacobi_Det jacobi_det(world_rank);
    Jacobi_Det_xi jacobi_det_xi(world_rank);
    const Physic_xi1 physic_xi1(world_rank);
    const Physic_xi2 physic_xi2(world_rank);
    //ProjectedSaver2D projected_saver_2D(dist_function,physic_xi1,physic_xi2,axis_vr,axis_vt,jacobi_det_xi);
    ProjectedSaver2D projected_saver_2D(dist_function,physic_vx,physic_vy,axis_vr,axis_vt,jacobi_det);

    init(world_rank,jacobi_det,dist_function);
    boundary_manager.apply<Axis_vr>();
    boundary_manager.apply<Axis_vt>();

    const Value dt = (Global::grid_size_vr*Global::grid_size_vt/4.)*0.1;

    const int num_steps = 40000;
    //const int num_steps = 0;
    const int save_span = 800;

    Timer timer;
    timer.start();

    static_assert(Axis_vt::num_blocks ==1);

    PolarCoordinates::SingularityProcessor s_lut(Axis_vt::num_global_grid / 2, Global::Vr::A);
    double phi_ = std::atan2(e_field.y, e_field.x);

    if (phi_ < 0.) {
        phi_ += 2.0 * M_PI;
    }
    double l_ = sqrt(e_field.x*e_field.x+e_field.y*e_field.y)*dt;
    s_lut.set_l_and_phi(l_, phi_);
    double sum = 0.;
    int k = 1;
    for(int i=k;i< Axis_vt::num_global_grid;i++){
        //sum += print(s_lut, k-1, i, e_field.x, dt);
    }
    for(int i=1;i<Axis_vt::num_global_grid;i++){
        for(int j=0;j<i;j++){
            print(s_lut, j, i);
        }
    }

    std::vector<double> buf(Axis_vt::num_global_grid);


    for(int i=0;i<num_steps;i++){
        if(i%1000 == 0 )std::cout<<i<<"\n";
        equation.solve<Axis_vr>(dt/2.);
        boundary_manager.apply<Axis_vr>();
        
        if(axis_vr.block_id == 0){
            for(int j=0;j<Axis_vt::num_global_grid;j++){
                buf[j]=dist_function.at(0,j);
            }
        }
        equation.solve<Axis_vt>(dt);
        
        if(axis_vr.block_id == 0){
            for(int j=0;j<Axis_vt::num_global_grid;j++){
                dist_function.at(0,j) = buf[j];
            }
        }
        if(axis_vr.block_id==0){
            for(int j=0;j<buf.size();j++)buf[j]=0.;
            double phi = std::atan2(e_field.y, e_field.x);

            if (phi < 0.) {
                phi += 2.0 * M_PI;
            }
            double l = sqrt(e_field.x*e_field.x+e_field.y*e_field.y)*dt;
            s_lut.set_l_and_phi(l, phi);
            for(int j=1;j<Axis_vt::num_global_grid;j++){
                for(int k=0;k<j;k++){
                    const Value area = s_lut.calc_area(k,j);
                    if(area>0.){
                        buf[j]+= dist_function.at(0,k) * area;
                        buf[k]-= dist_function.at(0,k) * area;
                    }
                    else{
                        buf[j]-= dist_function.at(0,j) * (-area);
                        buf[k]+= dist_function.at(0,j) * (-area);
                    }
                }
            }
            for(int j=0;j<Axis_vt::num_global_grid;j++){
                dist_function.at(0,j) += buf[j];
            }
        }
        boundary_manager.apply<Axis_vt>();
        equation.solve<Axis_vr>(dt/2.);
        boundary_manager.apply<Axis_vr>();

        if(i% save_span == 0){
            projected_saver_2D.save(
                "../output/0D2V_pole/step_" 
                + std::to_string(i/save_span) 
                + "_rank_" 
                + std::to_string(world_rank)
                + ".bin");
        }
    }
    timer.stop();
    if(world_rank == 0){
        std::cout<<timer<<"\n";
    }
    return 0;
}