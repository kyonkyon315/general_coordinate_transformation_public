#include <cmath>

#include "../include.h"


using Value = double;
using namespace std;

//計算空間の座標を設定します。
//Axis<ここには軸の通し番号をintで入力します。,ここには座標のグリッドの数をintで入力します,3,3>
//全体をなめる計算においては、通し番号が小さいものほど、より外側のループを担当することになります。
//また、x,v空間で、∂x/∂v = 0である必要があります。（電流の計算を簡単に行うための措置です。）
//∂v/∂x = 0 は要求されていません。（例えば背景磁場に沿って速度空間の向きを変えたい時など）
//通し番号は重複することなく、互いに隣り合った0以上の整数である必要があります。また、0を含む必要があります。
//計算空間の軸なので、一律Δx=1であり、軸同士は直交しています。
//最後の3,3 >はゴーストセルのグリッド数です。
//物理空間↔計算空間の写像は、全単射である必要があります。
using Axis_x_ = Axis<0,20,3,3>;
using Axis_vr = Axis<1,10,3,3>;
using Axis_vt = Axis<2,10,3,3>;
using Axis_vp = Axis<3,10,3,3>;

//電子分布関数の型を定義
//先頭に入力する型はテンソルの値の型です。その後に続く軸は、通し番号が小さいものほど左に入力してください。
using DistributionFunction = NdTensorWithGhostCell<Value,Axis_x_,Axis_vr,Axis_vt,Axis_vp>;

template<typename Element>
class Pair{
    public:
    using Element_t = Element;
    Element m_half;
    Element p_half;
};
//磁場の型を定義
using MagneticField = Pair<NdTensorWithGhostCell<Vec3<Value>,Axis_x_>>;
//B(i,j).p_half=B(x=Δx i,t=Δt(j+1/2))

//電場の型を定義
using ElectricField = NdTensorWithGhostCell<Vec3<Value>,Axis_x_>;
//E(i,j)=E(x=Δx(i+1/2),t=Δt j)

//電流の型を定義
using Current = NdTensorWithGhostCell<Vec3<Value>,Axis_x_>;
//current.at(i).z = j_z(x=Δx(i+1/2))
//current.at(i).x = j_x(x=Δx(i+1/2))
//current.at(i).y = j_y(x=Δx(i+1/2))

//電流計算が不要の時（磁場固定のときなど）はCurrentをNone_currentにしておく
//using Current = None_current;

//グローバル変数としてインスタンス化しておく。
namespace Global{
    DistributionFunction dist_function;
    ElectricField e_field;
    MagneticField m_field;
    Current current;
}

/***********************************************
 * 物理空間と計算空間の関係を表す関数を書きます(始)*
 ***********************************************/

// --- グローバル定数とヘルパー関数の定義 ---

constexpr Value grid_size_x_ = 3.;
const Value grid_size_vr = 4.;
const Value grid_size_vt = M_PI / (double)(Axis_vt::num_grid);
const Value grid_size_vp = 2.*M_PI / (double)(Axis_vp::num_grid);

Value vr(int calc_vr){ return grid_size_vr * (0.5 + (double)calc_vr);}
Value vt(int calc_vt){ return grid_size_vt * (0.5 + (double)calc_vt);}
Value vp(int calc_vp){ return grid_size_vp * (0.5 + (double)calc_vp);}


// --- 物理量クラス ---
//honestly_translateで計算座標↔物理座標の変換の式を定義します。
//それを用いてコンストラクタで各場所での値を事前計算してテーブルに格納します。（table.set_value(honestly_translate))
//シミュレーション中はテーブルを参照します。
//こちらも計算軸クラスと同様に通し番号を設定します。

class Physic_x_
{
public:
    Physic_x_(){}
    Value translate(int calc_x,int calc_vr,int calc_vt,int calc_vp)const{
        return grid_size_x_*calc_x;
    }
    static const int label = 0;
};

class Physic_vx
{
private:
    NdTensor<Value,Axis_vr,Axis_vt,Axis_vp> table;
    static Value honestly_translate(int calc_vr,int calc_vt,int calc_vp){
        // v_x = vr * sin(vt) * cos(vp)
        return vr(calc_vr) * sin(vt(calc_vt)) * cos(vp(calc_vp));
    }
public:
    Physic_vx(){table.set_value(honestly_translate);}
    Value translate(int calc_x_,int calc_vr,int calc_vt,int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);    
    }
    static const int label = 1;
};

class Physic_vy
{
private:
    NdTensor<Value,Axis_vr,Axis_vt,Axis_vp> table;
    static Value honestly_translate(int calc_vr,int calc_vt,int calc_vp){
        // v_y = vr * sin(vt) * sin(vp)
        return vr(calc_vr) * sin(vt(calc_vt)) * sin(vp(calc_vp));
    }
public:
    Physic_vy(){table.set_value(honestly_translate);}
    Value translate(int calc_x_,int calc_vr,int calc_vt,int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);    
    }
    static const int label = 2;
};

class Physic_vz
{
private:
    NdTensor<Value,Axis_vr,Axis_vt> table;
    static Value honestly_translate(int calc_vr,int calc_vt){
        // v_z = vr * cos(vt)
        return vr(calc_vr) * cos(vt(calc_vt));
    }
public:
    Physic_vz(){table.set_value(honestly_translate);}
    Value translate(int calc_x_,int calc_vr,int calc_vt,int calc_vp)const{
        return table.at(calc_vr,calc_vt);    
    }
    static const int label = 3;
};

//グローバル変数としてインスタンス化しておく。
namespace Global{
    Physic_x_ physic_x_;
    Physic_vx physic_vx;
    Physic_vy physic_vy;
    Physic_vz physic_vz;
}
/***********************************************
 * 計算軸を物理軸で微分した値の関数を書きます　(始)*
 ***********************************************/



class X__diff_x_
{
public:
    X__diff_x_(){}
    Value at(int calc_x_,int calc_vr,int calc_vt,int calc_vp)const{
        return 1./grid_size_x_;
    }
};

class Vr_diff_vx
{
private:
    NdTensor<Value,Axis_vt,Axis_vp> table;
    static Value honestly_calc(int calc_vt,int calc_vp){
        // d(vr) / d(vx) = sin(theta) * cos(phi)
        return sin(vt(calc_vt)) * cos(vp(calc_vp)) / grid_size_vr;
    }
public:
    Vr_diff_vx(){table.set_value(honestly_calc);}
    Value at(int calc_x_,int calc_vr,int calc_vt,int calc_vp)const{
        return table.at(calc_vt,calc_vp);
    }
};

class Vr_diff_vy
{
private:
    NdTensor<Value,Axis_vt,Axis_vp> table;
    static Value honestly_calc(int calc_vt,int calc_vp){
        // d(vr) / d(vy) = sin(theta) * sin(phi)
        return sin(vt(calc_vt)) * sin(vp(calc_vp)) / grid_size_vr;
    }
public:
    Vr_diff_vy(){table.set_value(honestly_calc);}
    Value at(int calc_x_,int calc_vr,int calc_vt,int calc_vp)const{
        return table.at(calc_vt,calc_vp);
    }
};

class Vr_diff_vz
{
private:
    NdTensor<Value,Axis_vt,Axis_vp> table;
    static Value honestly_calc(int calc_vt,int calc_vp){
        // d(vr) / d(vz) = cos(theta)
        return cos(vt(calc_vt)) / grid_size_vr;
    }
public:
    Vr_diff_vz(){table.set_value(honestly_calc);}
    Value at(int calc_x_,int calc_vr,int calc_vt,int calc_vp)const{
        return table.at(calc_vt,calc_vp);
    }
};

class Vt_diff_vx
{
private:
    NdTensor<Value,Axis_vr,Axis_vt,Axis_vp> table;
    static Value honestly_calc(int calc_vr,int calc_vt,int calc_vp){
        // d(v_theta) / d(vx) = cos(theta) * cos(phi) / vr
        return cos(vt(calc_vt)) * cos(vp(calc_vp)) / (vr(calc_vr) * grid_size_vt);
    }
public:
    Vt_diff_vx(){table.set_value(honestly_calc);}
    Value at(int calc_x_,int calc_vr,int calc_vt,int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);
    }
};

class Vt_diff_vy
{
private:
    NdTensor<Value,Axis_vr,Axis_vt,Axis_vp> table;
    static Value honestly_calc(int calc_vr,int calc_vt,int calc_vp){
        // d(v_theta) / d(vy) = cos(theta) * sin(phi) / vr
        return cos(vt(calc_vt)) * sin(vp(calc_vp)) / (vr(calc_vr) * grid_size_vt);
    }
public:
    Vt_diff_vy(){table.set_value(honestly_calc);}
    Value at(int calc_x_,int calc_vr,int calc_vt,int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);
    }
};

class Vt_diff_vz
{
private:
    NdTensor<Value,Axis_vr,Axis_vt,Axis_vp> table;
    static Value honestly_calc(int calc_vr,int calc_vt,int calc_vp){
        // d(v_theta) / d(vz) = -sin(theta) / vr
        return -sin(vt(calc_vt)) / (vr(calc_vr) * grid_size_vt);
    }
public:
    Vt_diff_vz(){table.set_value(honestly_calc);}
    Value at(int calc_x_,int calc_vr,int calc_vt,int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);
    }
};

class Vp_diff_vx
{
private:
    NdTensor<Value,Axis_vr,Axis_vt,Axis_vp> table;
    static Value honestly_calc(int calc_vr,int calc_vt,int calc_vp){
        // d(v_phi) / d(vx) = -sin(phi) / (vr * sin(theta))
        return -sin(vp(calc_vp)) / (vr(calc_vr) * sin(vt(calc_vt)) * grid_size_vp);
    }
public:
    Vp_diff_vx(){table.set_value(honestly_calc);}
    Value at(int calc_x_,int calc_vr,int calc_vt,int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);
    }
};

class Vp_diff_vy
{
private:
    NdTensor<Value,Axis_vr,Axis_vt,Axis_vp> table;
    static Value honestly_calc(int calc_vr,int calc_vt,int calc_vp){
        // d(v_phi) / d(vy) = cos(phi) / (vr * sin(theta))
        return cos(vp(calc_vp)) / (vr(calc_vr) * sin(vt(calc_vt)) * grid_size_vp);
    }
public:
    Vp_diff_vy(){table.set_value(honestly_calc);}
    Value at(int calc_x_,int calc_vr,int calc_vt,int calc_vp)const{
        return table.at(calc_vr,calc_vt,calc_vp);
    }
};

//グローバル変数としてインスタンス化
namespace Global{
    const Independent independent;
    const X__diff_x_ x__diff_x_;
    const Vr_diff_vx vr_diff_vx;
    const Vr_diff_vy vr_diff_vy;
    const Vr_diff_vz vr_diff_vz;
    const Vt_diff_vx vt_diff_vx;
    const Vt_diff_vy vt_diff_vy;
    const Vt_diff_vz vt_diff_vz;
    const Vp_diff_vx vp_diff_vx;
    const Vp_diff_vy vp_diff_vy;
}

/*******************************************************************
 * Jacobian行列を定義します。上で作成したクラスを行列風に並べてください。*
 * 互いに独立な軸の箇所（微分が０）はIndependent classを入れてください。*
 *                                                                 *
 * 具体的には、Jacobian[I,J]には「通し番号Iの計算軸」を「通し番号Jの物理*
 * 軸」で微分したものを入れてください。                               *
 *******************************************************************/

namespace Global{
Jacobian jacobian(
    x__diff_x_ , independent, independent, independent,
    independent, vr_diff_vx , vr_diff_vy , vr_diff_vz ,
    independent, vt_diff_vx , vt_diff_vy , vt_diff_vz ,
    independent, vp_diff_vx , vp_diff_vy , independent
);
}
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
public:
    Fx_(){}
    Value at(int calc_x_, int calc_vr, int calc_vt, int calc_vp) const {
        return Global::physic_vx.translate(calc_x_, calc_vr, calc_vt, calc_vp);
    }
};

//------------------------------------------
// 2. q/m (E + v×B)_x
//------------------------------------------
class Fvx {
private:
public:
    Fvx(){}

    Value at(int calc_x_, int calc_vr, int calc_vt, int calc_vp) const {
        const Value vx = Global::physic_vx.translate(calc_x_, calc_vr, calc_vt, calc_vp);
        const Value vy = Global::physic_vy.translate(calc_x_, calc_vr, calc_vt, calc_vp);
        const Value vz = Global::physic_vz.translate(calc_x_, calc_vr, calc_vt, calc_vp);

        //Yee格子を採用しているため、電場はｘ方向に、磁場はｔ方向に補間しなければならない。
        const Value Ex = (Global::e_field.at(calc_x_-1).x
                        + Global::e_field.at(calc_x_).x)/2.;
        const Value By = (Global::m_field.m_half.at(calc_x_).y
                        + Global::m_field.p_half.at(calc_x_).y)/2.;
        const Value Bz = (Global::m_field.m_half.at(calc_x_).z
                        + Global::m_field.p_half.at(calc_x_).z)/2.;
        return Parameters::Q/Parameters::m * (Ex + vy * Bz - vz * By);
    }
};

//------------------------------------------
// 3. q/m (E + v×B)_y
//------------------------------------------
class Fvy {

public:
    Fvy(){}

    Value at(int calc_x_, int calc_vr, int calc_vt, int calc_vp) const {
        const Value vx = Global::physic_vx.translate(calc_x_, calc_vr, calc_vt, calc_vp);
        const Value vy = Global::physic_vy.translate(calc_x_, calc_vr, calc_vt, calc_vp);
        const Value vz = Global::physic_vz.translate(calc_x_, calc_vr, calc_vt, calc_vp);

        //Yee格子を採用しているため、電場はｘ方向に、磁場はｔ方向に補間しなければならない。
        const Value Ey = (Global::e_field.at(calc_x_-1).y
                        + Global::e_field.at(calc_x_).y)/2.;
        const Value Bz = (Global::m_field.m_half.at(calc_x_).z
                        + Global::m_field.p_half.at(calc_x_).z)/2.;
        const Value Bx = (Global::m_field.m_half.at(calc_x_).x
                        + Global::m_field.p_half.at(calc_x_).x)/2.;
                        
        return Parameters::Q/Parameters::m * (Ey + vz * Bx - vx * Bz);
    }
};

//------------------------------------------
// 4. q/m (E + v×B)_z
//------------------------------------------
class Fvz {
public:
    Fvz(){}

    Value at(int calc_x_, int calc_vr, int calc_vt, int calc_vp) const {
        const Value vx = Global::physic_vx.translate(calc_x_, calc_vr, calc_vt, calc_vp);
        const Value vy = Global::physic_vy.translate(calc_x_, calc_vr, calc_vt, calc_vp);
        const Value vz = Global::physic_vz.translate(calc_x_, calc_vr, calc_vt, calc_vp);

        //Yee格子を採用しているため、電場はｘ方向に、磁場はｔ方向に補間しなければならない。
        const Value Ez = (Global::e_field.at(calc_x_-1).z
                        + Global::e_field.at(calc_x_).z)/2.;
        const Value Bx = (Global::m_field.m_half.at(calc_x_).x
                        + Global::m_field.p_half.at(calc_x_).x)/2.;
        const Value By = (Global::m_field.m_half.at(calc_x_).y
                        + Global::m_field.p_half.at(calc_x_).y)/2.;

        return Parameters::Q/Parameters::m * (Ez + vx * By - vy * Bx);
    }
};
namespace Global{
    Fx_ flux_x_;
    Fvx flux_vx;
    Fvy flux_vy;
    Fvz flux_vz;
}

/****************************************************************************
 * 次に、Flux計算機を選択します。今回は、Umeda2008を用います。
 ****************************************************************************/

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
 * 
 ****************************************************************************/
//xは周期境界条件
class BoundaryCondition_x_
{
public:
    static const int label = 0;
    template<typename Func>
    static Value left(Func func,int calc_x_, int calc_vr, int calc_vt, int calc_vp){
        return func(-calc_x_, calc_vr, calc_vt, calc_vp);
    }
    template<typename Func>
    static Value right(Func func,int calc_x_, int calc_vr, int calc_vt, int calc_vp){
        return func(calc_x_ - Axis_x_::num_grid, calc_vr, calc_vt, calc_vp);
    }
};

class BoundaryCondition_vr
{
public:
    static const int label = 1;
    template<typename Func>
    static Value left(Func func,int calc_x_, int calc_vr, int calc_vt, int calc_vp){
        static_assert(Axis_vp::num_grid%2 == 0,"v_phy空間のグリッド数は偶数である必要がある");
        constexpr int vp_half_num_grid=Axis_vp::num_grid/2;
        const int index_vp=(
            calc_vp < vp_half_num_grid ? 
            calc_vp+vp_half_num_grid
            :calc_vp-vp_half_num_grid
        );
        return func(calc_x_, -calc_vr-1, Axis_vt::num_grid-calc_vt,index_vp);
    }

    //物理的におかしいけど、とりあえずこうしておく
    template<typename Func>
    static Value right(Func func,int calc_x_, int calc_vr, int calc_vt, int calc_vp){
        return func(calc_x_, 2*Axis_vr::num_grid-calc_vr-1, calc_vt, calc_vp);
    }
};

// vt 
class BoundaryCondition_vt
{
public:
    static const int label = 2;
    template<typename Func>
    static Value left(Func func,int calc_x_, int calc_vr, int calc_vt, int calc_vp){
        static_assert(Axis_vp::num_grid%2 == 0,"v_phy空間のグリッド数は偶数である必要がある");
        constexpr int vp_half_num_grid=Axis_vp::num_grid/2;
        const int index_vp=(
            calc_vp < vp_half_num_grid ? 
            calc_vp+vp_half_num_grid
            :calc_vp-vp_half_num_grid
        );
        return func(calc_x_, calc_vr, -calc_vt-1, index_vp);
    }
    template<typename Func>
    static Value right(Func func,int calc_x_, int calc_vr, int calc_vt, int calc_vp){
        static_assert(Axis_vp::num_grid%2 == 0,"v_phy空間のグリッド数は偶数である必要がある");
        constexpr int vp_half_num_grid=Axis_vp::num_grid/2;
        const int index_vp=(
            calc_vp < vp_half_num_grid ? 
            calc_vp+vp_half_num_grid
            :calc_vp-vp_half_num_grid
        );
        return func(calc_x_, calc_vr, 2*Axis_vt::num_grid-calc_vt-1, index_vp);
    }
};

//vp 極座標におけるphy は当然周期境界条件

class BoundaryCondition_vp
{
public:
    static const int label = 3;
    template<typename Func>
    static Value left(Func func,int calc_x_, int calc_vr, int calc_vt, int calc_vp){
        return func(calc_x_, calc_vr, calc_vt, Axis_vp::num_grid-calc_vp);
    }
    template<typename Func>
    static Value right(Func func,int calc_x_, int calc_vr, int calc_vt, int calc_vp){
        return func(calc_x_, calc_vr, calc_vt, calc_vp- Axis_vp::num_grid);
    }
};
/*--------------------------------------
 * Pack を用いて境界条件をまとめます。
 *----------------------------------------------*/
namespace Global{
    BoundaryCondition_x_ boundary_condition_x_;
    BoundaryCondition_vr boundary_condition_vr;
    BoundaryCondition_vt boundary_condition_vt;
    BoundaryCondition_vp boundary_condition_vp;

    Pack boundary_condition(
        boundary_condition_x_,
        boundary_condition_vr,
        boundary_condition_vt,
        boundary_condition_vp
    );
}
/*----------------------------------------------------------------------------
 * ターゲットとなる関数とboundary_conditionを用いてboundary_managerを作成します。
 *---------------------------------------------------------------------------*/

namespace Global{
    BoundaryManager boundary_manager(dist_function,boundary_condition);
}


/****************************************************************************
 * 最後に、解くべき移流方程式を定義します。
 * 
 *物理演算子はOperatorsに、物理移流項はAdvectionsに、それぞれclass Packを用いてまとめる。
 *ただし、Operatorsの順番とAdvectionsの順番は式の順番と同じにしてください。
 *
 *さらにそれらOperatorsとAdvections、および発展させたい関数（ここではdist_func）を
 *用いてAdvectionEquationをインスタンス化します。これが、本シミュレーションのメインと
 *なるソルバーとして働きます。
 ****************************************************************************/

namespace Global{
    Pack operators(physic_x_,physic_vx,physic_vy,physic_vz);
    Pack advections(flux_x_,flux_vx,flux_vy,flux_vz);
    AdvectionEquation equation(dist_function,operators,advections,jacobian,scheme,boundary_condition,current);
}

/*
fdtd 関連
*/

namespace Global{
    Norm::Derived<Norm::SI> norm;
    FDTD_solver_1d fdtd_solver(e_field,m_field,current,norm);
    CalcCurrent_1d current_calculator(current,dist_function,operators);
}

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

int main(){
    Value dt = 0.1;
    int num_steps = 100;
    for(int i=0;i<num_steps;i++){
        std::cout<<i<<std::endl;
        Global::equation.solve<Axis_vr>(dt/2.);
        Global::boundary_manager.apply<Axis_vr>();
        Global::equation.solve<Axis_vt>(dt/2.);
        Global::boundary_manager.apply<Axis_vt>();
        Global::equation.solve<Axis_vp>(dt/2.);
        Global::boundary_manager.apply<Axis_vp>();
        Global::equation.solve<Axis_x_>(dt/2.);
        Global::boundary_manager.apply<Axis_x_>();

        Global::current_calculator.calc();
        Global::fdtd_solver.develop(dt / grid_size_x_);
        apply_periodic_1d(Global::e_field);
        apply_periodic_1d(Global::m_field.p_half);
        apply_periodic_1d(Global::m_field.m_half);

        Global::equation.solve<Axis_x_>(dt/2.);
        Global::boundary_manager.apply<Axis_x_>();
        Global::equation.solve<Axis_vp>(dt/2.);
        Global::boundary_manager.apply<Axis_vp>();
        Global::equation.solve<Axis_vt>(dt/2.);
        Global::boundary_manager.apply<Axis_vt>();
        Global::equation.solve<Axis_vr>(dt/2.);
        Global::boundary_manager.apply<Axis_vr>();
    }
    return 0;
}