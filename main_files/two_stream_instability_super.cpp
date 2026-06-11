#include "mpi.h"
#include <cmath>
#include <csignal>
#include <random>

//#include "../include_super.h"
//#include "../supercomputer_instruments/axis_instantiator.h"
int rank;

using Value = double;
using namespace std;

//計算空間の座標を設定します。
//Axis<ここには軸の通し番号をintで入力します。,
// ここには座標のグリッドの数をintで入力します,並列化で何個に分けるか,ゴーストセル数>
//全体をなめる計算においては、通し番号が小さいものほど、より外側のループを担当することになります。
//また、x,v空間で、∂x/∂v = 0である必要があります。（電流の計算を簡単に行うための措置です。）
//∂v/∂x = 0 は要求されていません。（例えば背景磁場に沿って速度空間の向きを変えたい時など）
//通し番号は重複することなく、互いに隣り合った0以上の整数である必要があります。また、0を含む必要があります。
//計算空間の軸なので、一律Δx=1であり、軸同士は直交しています。
//物理空間↔計算空間の写像は、全単射である必要があります。
#include "../supercomputer_instruments/axis.h"
#include "../supercomputer_instruments/axis_instantiator.h"
using Axis_x_ = Axis<0,256/4,4,3>;
using Axis_vx = Axis<1,512/4,4,3>;

//電子分布関数の型を定義
//先頭に入力する型はテンソルの値の型です。その後に続く軸は、通し番号が小さいものほど左に入力してください。
//入れる軸の数を変えることで次元数を調節できます。
#include "../supercomputer_instruments/n_d_tensor_with_ghost_cell.h"
using DistributionFunction = NdTensorWithGhostCell<Value,Axis_x_,Axis_vx>;

//磁場の型を定義
#include "../vec3.h"
using MagneticField = NdTensorWithGhostCell<Vec3<Value>,Axis_x_>;
//B(i,j)=B(x=Δx i,t=Δt(j+1/2))

//電場の型を定義
using ElectricField = NdTensorWithGhostCell<Vec3<Value>,Axis_x_>;
//E(i,j)=E(x=Δx(i+1/2),t=Δt j)

#include "../pack.h"
using VeloPack = Pack<Axis_vx>;
//電流の型を定義
//電流は実空間のみのグリッドを持つので、Axis_vxは与えない。
//ただし、電流計算用の足し合わせで速度空間の情報が必要なので、Pack<Axis_vx>を与える。
#include "../supercomputer_instruments/current.h"
using Current_type = Current<Vec3<Value>,VeloPack,Axis_x_>;
//current.at(i).x = j_x(x=Δx(i+1/2))
//current.at(i).y = j_y(x=Δx(i+1/2))
//current.at(i).z = j_z(x=Δx(i+1/2))

//電流計算が不要の時（磁場固定のときなど）はCurrentをNone_currentにしておく
//using Current = None_current;


/***********************************************
 * 物理空間と計算空間の関係を表す関数を書きます(始)*
 ***********************************************/
#include "../normalization.h"
// --- グローバル定数とヘルパー関数の定義 ---
namespace Global{
    constexpr Value grid_size_x_ = 0.5*3.3;
    //0.3 * lambda_D

    constexpr Value v_max = 5.*3.3* Norm::Param::v_thermal/Norm::Base::v0;
    constexpr Value grid_size_vx = 2. * v_max / Axis_vx::num_grid;
}//Global

// --- 物理量クラス ---
//honestly_translateで計算座標↔物理座標の変換の式を定義します。
//それを用いてコンストラクタで各場所での値を事前計算してテーブルに格納します。（table.set_value(honestly_translate))
//シミュレーション中はテーブルを参照します。
//こちらも計算軸クラスと同様に通し番号を設定します。

//計算空間はグリッドサイズが１なので、それを意味のあるスケールに変換するクラスをつくります
class CalcX__2_X_{
private:
    const int x__start_id;
    static int calc_start_id(const int my_world_rank){
        auto [axis_x_, axis_vx] = axis_instantiator<Axis_x_,Axis_vx>(my_world_rank);
        return axis_x_.L_id;
    }
public:
    CalcX__2_X_(const int my_world_rank):
        x__start_id(calc_start_id(my_world_rank))
    {}

    Value apply(const int calc_x_)const{ return Global::grid_size_x_ * (0.5 + (double)(x__start_id + calc_x_));}
};

class CalcVx_2_Vx{
private:
    const int vx_start_id;
    static int calc_start_id(const int my_world_rank){
        auto [axis_x_, axis_vx] = axis_instantiator<Axis_x_,Axis_vx>(my_world_rank);
        return axis_vx.L_id;
    }
public:
    CalcVx_2_Vx(const int my_world_rank):
        vx_start_id(calc_start_id(my_world_rank))
    {}

    Value apply(const int calc_vx)const{ return Global::grid_size_vx * (0.5 + (double)(vx_start_id + calc_vx) - 0.5 * (Value)Axis_vx::num_global_grid);}
};

class Physic_x_
{
    const CalcX__2_X_ calc_x__2__x;
public:
    Physic_x_(const int my_world_rank):
        calc_x__2__x(my_world_rank)
    {}
    Value honestly_translate(int calc_x_,int calc_vx)const{
        return calc_x__2__x.apply(calc_x_);
    }
    Value translate(int calc_x_,int calc_vx)const{
        return calc_x__2__x.apply(calc_vx);
    }
    static const int label = 0;
};

class Physic_vx
{
    const CalcVx_2_Vx calc_vx_2_vx;
public:
    Physic_vx(const int my_world_rank):
        calc_vx_2_vx(my_world_rank)
    {}
    Value honestly_translate(int calc_x_,int calc_vx)const{
        return calc_vx_2_vx.apply(calc_vx);    
    }
    Value translate(int calc_x_,int calc_vx)const{
        return calc_vx_2_vx.apply(calc_vx);    
    }
    static const int label = 1;
};

/***********************************************
 * 計算軸を物理軸で微分した値の関数を書きます　(始)*
 ***********************************************/

class X__diff_x_
{
public:
    X__diff_x_(){}
    Value at(int calc_x_,int calc_vx)const{
        return 1./Global::grid_size_x_;
    }
};

class Vx_diff_vx
{
public:
    Vx_diff_vx(){}
    Value at(int calc_x_,int calc_vx)const{
        return 1./Global::grid_size_vx;
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
public:
    Value at(const int calc_x_,const int calc_vx)const{
        return Global::grid_size_vx*Global::grid_size_x_;
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
class Fx_ {
private:
    const Physic_vx physic_vx;
public:
    Fx_(const int my_world_rank):
        physic_vx(my_world_rank)
    {}
    Value at(int calc_x_, int calc_vx) const {
        return physic_vx.translate(calc_x_, calc_vx);
    }
};

bool is_velo_left_edge(const int my_world_rank){
    auto [axis_x_, axis_vx] = axis_instantiator<Axis_x_,Axis_vx>(my_world_rank);
    return axis_vx.block_id == 0;
}

bool is_velo_right_edge(const int my_world_rank){
    auto [axis_x_, axis_vx] = axis_instantiator<Axis_x_,Axis_vx>(my_world_rank);
    return axis_vx.block_id == Axis_vx::num_blocks-1;
}

class Fvx {
private:
    const ElectricField& e_field;
    const bool _is_velo_right_edge;
    const bool _is_velo_left_edge;
public:
    Fvx(const int my_world_rank,
        const ElectricField& e_field
    ):
        e_field(e_field),
        _is_velo_left_edge(is_velo_left_edge(my_world_rank)),
        _is_velo_right_edge(is_velo_right_edge(my_world_rank))
    {}
    Value at(int calc_x_, int calc_vx) const {
        if (_is_velo_left_edge && calc_vx == -1){
            return - at(calc_x_, 0);
        }
        else if(_is_velo_right_edge && calc_vx == Axis_vx::num_grid){
            return - at(calc_x_, Axis_vx::num_grid-1);
        }
        else{
            //Yee格子を採用しているため、電場はｘ方向に、磁場はｔ方向に補間しなければならない。
            const Value Ex = (e_field.at(calc_x_-1).z
                            + e_field.at(calc_x_).z)/2.;
            return -Ex;//電子の電荷が負なので - がつく
            //return Parameters::Q/Parameters::m * Ex;
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
class BoundaryCondition_x_
{
public:
    static const int label = 0;

    //吸収項などを実装するとき、ゴーストセルにほかのセルの値を代入するだけではなくなる。このときtrueにする。->その場合の動作は未定義
    static constexpr bool not_only_comm = false;

    //ghost_cell[calc_x,calc_vx] <- cell[calc_x_ + Axis_x_::num_grid,calc_vx] 
    template<int Index>
    static int left(int calc_x_, int calc_vx){
        return 0;
    }
    //ghost_cell[calc_x,calc_vx] <- cell[calc_x_ - Axis_x_::num_grid,calc_vx] 
    template<int Index>
    static int right(int calc_x_, int calc_vx){
        return 0;
    }
};

template<>
inline int BoundaryCondition_x_::left<0>(const int calc_x_, const int calc_vx){
    return calc_x_ + Axis_x_::num_global_grid;
}
template<>
inline int BoundaryCondition_x_::left<1>(const int calc_x_, const int calc_vx){
    return calc_vx;
}

template<>
inline int BoundaryCondition_x_::right<0>(const int calc_x_, const int calc_vx){
    return calc_x_ - Axis_x_::num_global_grid;
}
template<>
inline int BoundaryCondition_x_::right<1>(const int calc_x_, const int calc_vx){
    return calc_vx;
    }
class BoundaryCondition_vx
{
public:
    static const int label = 1;
    static constexpr bool not_only_comm = false;

    template<int Index>
    static int left(int calc_x_, int calc_vx){
        return 0;
    }

    template<int Index>
    static int right(int calc_x_, int calc_vx){
        return 0;
    }
};


template<>
inline int BoundaryCondition_vx::left<0>(const int calc_x_, const int calc_vx){
    return calc_x_;
}
template<>
inline int BoundaryCondition_vx::left<1>(const int calc_x_, const int calc_vx){
    return - calc_vx - 1;
}

template<>
inline int BoundaryCondition_vx::right<0>(const int calc_x_, const int calc_vx){
    return calc_x_;
}
template<>
inline int BoundaryCondition_vx::right<1>(const int calc_x_, const int calc_vx){
    return Axis_vx::num_global_grid - 1 - (calc_vx - Axis_vx::num_global_grid);
}


/*--------------------------------------
 * Pack を用いて境界条件をまとめます。
 *----------------------------------------------*/
using BoundaryCondition = Pack<BoundaryCondition_x_, BoundaryCondition_vx>;
namespace Global{
    BoundaryCondition_x_ boundary_condition_x_;
    BoundaryCondition_vx boundary_condition_vx;

    Pack boundary_condition(
        boundary_condition_x_,
        boundary_condition_vx
    );
}

//電場、磁場についても境界条件をまとめる
class BoundaryCondition_EM_x_
{
public:
    static const int label = 0;

    //吸収項などを実装するとき、ゴーストセルにほかのセルの値を代入するだけではなくなる。このときtrueにする。->その場合の動作は未定義
    static constexpr bool not_only_comm = false;

    //ghost_cell[calc_x,calc_vx] <- cell[calc_x_ + Axis_x_::num_grid,calc_vx] 
    template<int Index>
    static int left(int calc_x_){
        return 0;
    }
    //ghost_cell[calc_x,calc_vx] <- cell[calc_x_ - Axis_x_::num_grid,calc_vx] 
    template<int Index>
    static int right(int calc_x_){
        return 0;
    }
};

template<>
inline int BoundaryCondition_EM_x_::left<0>(const int calc_x_){
    return calc_x_ + Axis_x_::num_global_grid;
}

template<>
inline int BoundaryCondition_EM_x_::right<0>(const int calc_x_){
    return calc_x_ - Axis_x_::num_global_grid;
}

using BoundaryCondition_EM = Pack<BoundaryCondition_EM_x_>;
namespace Global{
    BoundaryCondition_EM_x_ boundary_condition_EM_x_;

    Pack boundary_condition_em(
        boundary_condition_EM_x_
    );
}

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


template<typename Field>
void apply_periodic_1d(Field& f)
{
    constexpr int N  = Axis_x_::num_grid;
    constexpr int GL = Axis_x_::L_ghost_length;
    constexpr int GR = Axis_x_::R_ghost_length;

    for(int g = 1; g <= GL; ++g){
        f.at(-g) = f.at(N - g);
    }
    for(int g = 0; g < GR; ++g){
        f.at(N + g) = f.at(g);
    }
}
/*
fdtd関連の設定終わり。
*/

/*分布関数の初期化関数の設定*/
Value fM(Value v_tilde/*無次元量が入る*/){
    const Value U = 3.3 * Norm::Param::v_thermal / Norm::Base::v0;
    return Norm::Coef::Ne_tilde * std::exp(-(v_tilde-U)*(v_tilde-U)/2.)
           / Utils::ConstExpr::sqrt(2*M_PI)/2.
           + Norm::Coef::Ne_tilde * std::exp(-(v_tilde+U)*(v_tilde+U)/2.)
           / Utils::ConstExpr::sqrt(2*M_PI)/2.;
    //Ne_tilde = int f_tilde dv_tilde^3
}


void initialize_distribution(
    const int my_world_rank,
    DistributionFunction& f,
    int seed
)
{
    constexpr Value eps = 1e-3;
    Physic_vx physic_vx(my_world_rank);
    Jacobi_Det jacobi_det;

    std::mt19937 rng(12345 + seed);
    std::uniform_real_distribution<Value> uni(-1.0,1.0);

    for(int ix=0; ix<Axis_x_::num_grid; ix++){
        Value eta = uni(rng);  // x 依存ノイズ
        //Value eta = std::sin(30.*2.*M_PI*(Value)ix/(Value)Axis_x_::num_grid);
        Value base = 1.;
        //if(ix>Axis_x_::num_grid/4 && ix<3*Axis_x_::num_grid/4)base = 0.01;

        for(int iv=0; iv<Axis_vx::num_grid; iv++){
            Value v_tilde = physic_vx.translate(ix,iv);

            f.at(ix,iv)
                = jacobi_det.at(ix,iv)*fM(v_tilde) * (base + eps * eta);
                //やこびあんで計算空間にスケールする
        }
    }
}

void solve_poisson_1d_periodic(
    DistributionFunction& f,
    ElectricField& e_field
) 
{
    int N = Axis_x_::num_grid;
    
    // イオン密度を計算（電子密度の平均値
    Value n_e_tilde_avg = 0.0;
    for(int i=0;i<Axis_x_::num_grid;++i){
        for(int j=0;j<Axis_vx::num_grid;++j){
            n_e_tilde_avg += f.at(i,j);
        }
    }
    n_e_tilde_avg /= (Value)N;

    // 電場の積分
    e_field.at(0).z = 0.0; // 基準値（ポテンシャル自由度）
    for(int i=0;i<Axis_x_::num_grid;i++){
        Value n_e_tilde=0.;
        for(int j=0;j<Axis_vx::num_grid;++j){
            n_e_tilde += f.at(i,j);
        }
        e_field.at(i+1).z 
            = e_field.at(i).z 
            + Norm::Coef::poisson_coef * (n_e_tilde_avg - n_e_tilde)/* *gridsize(=1)*/;

    }
    
    // 平均を引いて、周期境界条件を調整
    double E_mean = 0.0;
    for(int i=0;i<N;i++) E_mean += e_field.at(i).z;
    E_mean /= (N);

    for(int i=0;i<N;i++) e_field.at(i).z -= E_mean;

    apply_periodic_1d(e_field);
}

#include "../supercomputer_instruments/advection_equation.h"
#include "../supercomputer_instruments/FDTD/fdtd_solver_1d.h"
#include "../supercomputer_instruments/axis_instantiator.h"
#include "../supercomputer_instruments/boundary_manager.h"
#include "../projected_saver_2D.hpp"
#include "../utils/Timer.h"

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
    
    const Physic_x_ physic_x_(world_rank);
    const Physic_vx physic_vx(world_rank);

    using Operators = Pack<Physic_x_,Physic_vx>;

    Operators operators(physic_x_,physic_vx);

    const Independent independent;
    const X__diff_x_ x__diff_x_;
    const Vx_diff_vx vx_diff_vx;

    const Jacobian jacobian(
        x__diff_x_ , independent, 
        independent, vx_diff_vx 
    );

    Fx_ flux_x_(world_rank);
    Fvx flux_vx(world_rank,e_field);

    const BoundaryCondition_x_ boundary_condition_x_;
    const BoundaryCondition_vx boundary_condition_vx;

    const Pack boundary_condition(
        boundary_condition_x_,
        boundary_condition_vx
    );

    const BoundaryCondition_EM_x_ boundary_condition_EM_x;
    const Pack boundary_condition_em(boundary_condition_EM_x);

    Pack advections(flux_x_, flux_vx);
    AdvectionEquation equation(world_rank,dist_function,advections,jacobian,Global::scheme, current);
    
    FDTD_solver_1d fdtd_solver(e_field,m_field,current);

    auto [axis_x_, axis_vx] = axis_instantiator<Axis_x_,Axis_vx>(world_rank);
   
    BoundaryManager boundary_manager(world_rank,world_size,dist_function,boundary_condition,axis_x_,axis_vx);
    BoundaryManager boundary_manager_e(world_rank,world_size,e_field,boundary_condition_em,axis_x_,axis_vx);
    BoundaryManager boundary_manager_m(world_rank,world_size,m_field,boundary_condition_em,axis_x_,axis_vx);
    
    
    //初期化
    initialize_distribution(world_rank,dist_function,axis_x_.block_id);
    // 初期化後のゴーストセル更新（重要）
    boundary_manager.apply<Axis_x_>();
    boundary_manager.apply<Axis_vx>();
    boundary_manager_e.apply<Axis_x_>();
    boundary_manager_m.apply<Axis_x_>();

    solve_poisson_1d_periodic(dist_function,e_field);

    Value dt = 0.01 ;

    int num_steps = 100000;
    std::ofstream ex_log("Ex_t.dat");
    std::ofstream f_log("f.dat");

    Jacobi_Det jacobi_det;
    ProjectedSaver2D projected_saver(
        dist_function,
        physic_x_,
        physic_vx,
        axis_x_,axis_vx,jacobi_det);
    Timer timer;

    timer.start();
    for(int i=0;i<num_steps;i++){
        //if(world_rank==0 && i%100==0)std::cout<<i<<std::endl;
        //v(0), x(0), E(0), B(1/2), J(-1/2)
        
        equation.solve<Axis_vx>(dt/2.);
        boundary_manager.apply<Axis_vx>();

        current.clear();

        //v(1/2), x(0), E(0), B(1/2), J(-1/2)

        equation.solve<Axis_x_>(dt);
        boundary_manager.apply<Axis_x_>();
        //Global::current_calculator.calc();
        current.compute_global_current();

        //v(1/2), x(1), E(0), B(1/2), J(1/2)

        fdtd_solver.develop_e(dt , Global::grid_size_x_);
        boundary_manager_e.apply<Axis_x_>();

        //v(1/2), x(1), E(1), B(1/2), J(1/2)
        

        equation.solve<Axis_vx>(dt/2.);
        boundary_manager.apply<Axis_vx>();
        //v(1), x(1), E(1), B(1/2), J(1/2)
        
        fdtd_solver.develop_m(dt , Global::grid_size_x_);
        boundary_manager_m.apply<Axis_x_>();
        //v(1), x(1), E(1), B(3/2), J(1/2)

        /*
        //if(i%20 == 0){
        if(false){
            dist_function.save_physical_fast("../output/two_stream/rank_" 
                                    + std::to_string(world_rank) 
                                    + "__step_" 
                                    + std::to_string(i/20) 
                                    + ".bin");
        }

        if(i%20 == 0){
        //if(false){
            projected_saver.save("../output/two_stream/rank_" 
                                    + std::to_string(world_rank) 
                                    + "__step_" 
                                    + std::to_string(i/20) 
                                    + ".bin");

            for(int ix=0; ix<Axis_x_::num_grid; ix++){
                ex_log << e_field.at(ix).z << " ";
                Value f = 0.;
                for(int j=0;j<Axis_vx::num_grid;j++){
                    f += dist_function.at(ix,j)/jacobi_det.at(ix,j);
                    //やこびあんで物理空間にスケールする
                }
                if(ix==3522)std::cout<<f<<" "<<std::endl;

                f_log << f <<" ";
            }
            ex_log << "\n";
            f_log << "\n";
        }
        */
    }
    timer.stop();
    if(world_rank==0)std::cout<<timer<<"\n";
    MPI_Finalize();
    return 0;
}