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
using Axis_vr = Axis<0,50,1,3>;
using Axis_vt = Axis<1,50,1,3>;

#include "../supercomputer_instruments/n_d_tensor_with_ghost_cell.h"
//電子分布関数の型を定義
//先頭に入力する型はテンソルの値の型です。その後に続く軸は、通し番号が小さいものほど左に入力してください。
using DistributionFunction = NdTensorWithGhostCell<Value,Axis_vr,Axis_vt>;

#include "../vec3.h"
//磁場の型を定義
using MagneticField = Vec3<Value>;

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
}
#include "../supercomputer_instruments/axis_instantiator.h"
//計算空間はグリッドサイズが１なので、それを意味のあるスケールに変換するクラスをつくります
class CalcVr_2_Vr{
private:
    const int vr_start_id;
    static int calc_start_id(const int my_world_rank){
        auto [axis_vr, axis_vt] = axis_instantiator<Axis_vr,Axis_vt>(my_world_rank);
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
        auto [axis_vr, axis_vt] = axis_instantiator<Axis_vr,Axis_vt>(my_world_rank);
        return axis_vt.L_id;
    }
public:
    CalcVt_2_Vt(const int my_world_rank):
        vt_start_id(calc_start_id(my_world_rank))
    {}
    Value apply(const int calc_vt)const{ return Global::grid_size_vt * (0.5 + (double)(vt_start_id+calc_vt));}
};


// --- 物理座標クラス ---
//honestly_translateで計算座標↔物理座標の変換の式を定義します。
//それを用いてコンストラクタで各場所での値を事前計算してテーブルに格納します。（table.set_value(honestly_translate))
//シミュレーション中はテーブルを参照します。
//こちらも計算軸クラスと同様に通し番号を設定します。
using FullSliceGhost_r = Slice<-Axis_vr::L_ghost_length, Axis_vr::num_grid+Axis_vr::R_ghost_length>;
using FullSliceGhost_t = Slice<-Axis_vt::L_ghost_length, Axis_vt::num_grid+Axis_vt::R_ghost_length>;

class Physic_vx
{
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
public:
    Value honestly_translate(int calc_vr,int calc_vt)const{
        // v_x = vr * cos(vt)
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        return vr * cos(vt);
    }

    Physic_vx(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            [this](const int calc_vr,const int calc_vt){return honestly_translate(calc_vr, calc_vt);}
        );
    }

    Value translate(int calc_vr,int calc_vt)const{
        return table.at(calc_vr,calc_vt);    
    }
    static const int label = 0;
};

class Physic_vy
{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt> table;
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
public:
    Value honestly_translate(int calc_vr,int calc_vt)const{
        // v_y = vr * sin(vt) 
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        return vr * sin(vt);
    }
    Physic_vy(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            [this](const int calc_vr,const int calc_vt){return honestly_translate(calc_vr, calc_vt);}
        );
    }
    Value translate(int calc_vr,int calc_vt)const{
        return table.at(calc_vr,calc_vt);    
    }
    static const int label = 1;
};

/***********************************************
 * 計算軸を物理軸で微分した値の関数を書きます　(始)*
 ***********************************************/

class Vr_diff_vx
{
private:
    NdTensorWithGhostCell<Value,Axis_vr,Axis_vt> table;
    const CalcVt_2_Vt calc_vt_2_vt;
    Value honestly_translate(const int calc_vr,const int calc_vt){
        // v_y = vr * sin(vt) 
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        return std::cos(vt)/(double)Global::grid_size_vr;
    }
public:
    Vr_diff_vx(const int my_world_rank):
        table(my_world_rank),
        calc_vt_2_vt(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            [this](const int calc_vr,const int calc_vt){return honestly_translate(calc_vr, calc_vt);}
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
    const CalcVt_2_Vt calc_vt_2_vt;
    Value honestly_translate(const int calc_vr,const int calc_vt){
        // v_y = vr * sin(vt) 
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        return std::sin(vt)/(double)Global::grid_size_vr;
    }
public:
    Vr_diff_vy(const int my_world_rank):
        table(my_world_rank),
        calc_vt_2_vt(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            [this](const int calc_vr,const int calc_vt){return honestly_translate(calc_vr, calc_vt);}
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
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
    Value honestly_translate(const int calc_vr,const int calc_vt){
        // v_y = vr * sin(vt) 
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        return  - std::sin(vt)/(vr*(double)Global::grid_size_vt);
    }
public:
    Vt_diff_vx(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            [this](const int calc_vr,const int calc_vt){return honestly_translate(calc_vr, calc_vt);}
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
    const CalcVr_2_Vr calc_vr_2_vr;
    const CalcVt_2_Vt calc_vt_2_vt;
    Value honestly_translate(const int calc_vr,const int calc_vt){
        // v_y = vr * sin(vt) 
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        const Value vt = calc_vt_2_vt.apply(calc_vt);
        return  std::cos(vt)/(vr*(double)Global::grid_size_vt);
    }
public:
    Vt_diff_vy(const int my_world_rank):
        table(my_world_rank),
        calc_vr_2_vr(my_world_rank),
        calc_vt_2_vt(my_world_rank)
    {
        table.set_value_sliced<FullSliceGhost_r,FullSliceGhost_t>(
            [this](const int calc_vr,const int calc_vt){return honestly_translate(calc_vr, calc_vt);}
        );
    }
    Value at(int calc_vr,int calc_vt)const{
        return table.at(calc_vr,calc_vt);
    }
};

class Jacobi_Det{
private:
    const CalcVr_2_Vr calc_vr_2_vr;
public:
    Jacobi_Det(const int my_world_rank):
        calc_vr_2_vr(my_world_rank)
    {}

    Value at(const int calc_vr,const int calc_vt)const{
        const Value vr = calc_vr_2_vr.apply(calc_vr);
        return Global::grid_size_vr*Global::grid_size_vt*vr;
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
    const bool _is_velo_right_edge;
    const MagneticField& m_field;
    const Physic_vy& physic_vy;
public:
    Fvx(const int my_world_rank,
        const MagneticField& m_field,
        const Physic_vy& physic_vy
    ):
        _is_velo_right_edge(is_velo_right_edge(my_world_rank)),
        m_field(m_field),
        physic_vy(physic_vy)
    {}
    Value at(int calc_vr, int calc_vt) const {
        const Value vy = physic_vy.translate(calc_vr, calc_vt);
        return - vy*m_field.z;//電子の電荷が負なので - がつく
    }
};

//------------------------------------------
// 2. q/m (E + v×B)_x
//------------------------------------------
class Fvy {
private:
    const bool _is_velo_right_edge;
    const MagneticField& m_field;
    const Physic_vx& physic_vx;
public:
    Fvy(const int my_world_rank,
        const MagneticField& m_field,
        const Physic_vx& physic_vx
    ):
        _is_velo_right_edge(is_velo_right_edge(my_world_rank)),
        m_field(m_field),
        physic_vx(physic_vx)
    {}
    Value at(int calc_vr, int calc_vt) const {
        const Value vx = physic_vx.translate(calc_vr, calc_vt);
        return vx*m_field.z;
    }
};
/****************************************************************************
 * 次に、Flux計算機を選択します。今回は、Umeda2008を用います。
 ****************************************************************************/
#include "../schemes/umeda_2008_fifth_order.h"
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
        else return 0;
    }

    template<int Index>
    static int right(const int calc_vr,const int calc_vt){
        if constexpr(Index == 0){
            return Axis_vr::num_global_grid - 1 - (calc_vr - Axis_vr::num_global_grid);
        }
        else if constexpr(Index == 1){
            return calc_vt;
        }
        else return 0;
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

Value fM(const Value vr_tilde/*無次元量が入る*/){
    return Norm::Coef::Ne_tilde * std::exp(-vr_tilde * vr_tilde /2.)
    //return std::exp(-vr_tilde * vr_tilde /2.)
           /(2.* M_PI );
    //Ne_tilde = int f_tilde dv_tilde^3
}

void init(int my_world_rank,const Jacobi_Det& jacobi_det,DistributionFunction& dist_function){
    CalcVr_2_Vr calc_vr_2_vr(my_world_rank);
    CalcVt_2_Vt calc_vt_2_vt(my_world_rank);
    for(int i=0;i<Axis_vr::num_grid;i++){
        for(int j=0;j<Axis_vt::num_grid;j++){
            const Value vr = calc_vr_2_vr.apply(i);
            dist_function.at(i,j) = fM(vr)*jacobi_det.at(i, j);
        }
    }
}

#include "../supercomputer_instruments/advection_equation.h"
#include "../supercomputer_instruments/FDTD/fdtd_solver_1d.h"
#include "../supercomputer_instruments/boundary_manager.h"
#include "../jacobian.h"

#include "../projected_saver_2D.hpp"
#include "../utils/Timer.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    DistributionFunction dist_function(world_rank);
    MagneticField m_field(world_rank);
    Current_type current;

    m_field.z = Norm::Coef::B_tilde;

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

    Fvx flux_vx(world_rank,m_field,physic_vy);
    Fvy flux_vy(world_rank,m_field,physic_vx);

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
    ProjectedSaver2D projected_saver_2D(dist_function,physic_vx,physic_vy,axis_vr,axis_vt,jacobi_det);

    init(world_rank,jacobi_det,dist_function);
    boundary_manager.apply<Axis_vr>();
    boundary_manager.apply<Axis_vt>();


    const Value courant_val = 0.3;
    const Value dt = courant_val * 2. * M_PI / (double)Axis_vt::num_global_grid;

    const int num_steps = (int)(2.*M_PI * 10000./dt);
    const int save_span = num_steps / 100;

    Timer timer;
    timer.start();
    for(int i=0;i<num_steps;i++){
        if(i%1000 == 0 )std::cout<<i<<"\n";
        equation.solve<Axis_vr>(dt/2.);
        boundary_manager.apply<Axis_vr>();
        equation.solve<Axis_vt>(dt);
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