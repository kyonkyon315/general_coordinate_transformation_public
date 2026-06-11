#include <cmath>
#include "../include.h"
const double Q = 1.602e-19;
const double m = 9.101e-31;

using Value = double;
using namespace std;

//計算空間の座標を設定します。
//Axis<ここには軸の通し番号をintで入力します。,ここには座標のグリッドの数をintで入力します,3,3>
//全体をなめる計算においては、通し番号が小さいものほど、より外側のループを担当することになります。
//通し番号は重複することなく、互いに隣り合った0以上の整数である必要があります。また、0を含む必要があります。
//計算空間の軸なので、一律Δx=1であり、軸同士は直交しています。
//最後の3,3 >はゴーストセルのグリッド数です。
using Axis_x_ = Axis<0,100,3,3>;
using Axis_vx = Axis<1,200,3,3>;
using Axis_vy = Axis<2,200,3,3>;

//電子分布関数の型を定義
//先頭に入力する型はテンソルの値の型です。その後に続く軸は、通し番号が小さいものほど左に入力してください。
using DistributionFunction = NdTensorWithGhostCell<Value,Axis_x_,Axis_vx,Axis_vy>;

//磁場の型を定義
using MagneticField = Vec3<Value>;
using ElectricField = Vec3<Value>;

//グローバル変数としてインスタンス化しておく。
namespace Global{
    DistributionFunction dist_function;
    MagneticField m_field;
    ElectricField e_field;
}

/***********************************************
 * 物理空間と計算空間の関係を表す関数を書きます(始)*
 ***********************************************/

// --- グローバル定数とヘルパー関数の定義 ---
const Value grid_size_x_ = 2.;
const Value grid_size_vx = 0.6;
const Value grid_size_vy = 0.6;

Value x_(int calc_x_){ return grid_size_x_ * (double)calc_x_;}
Value vx(int calc_vx){ return grid_size_vx * (double)(calc_vx-Axis_vx::num_grid/2);}
Value vy(int calc_vy){ return grid_size_vy * (double)(calc_vy-Axis_vy::num_grid/2);}



// --- 物理座標クラス ---
//honestly_translateで計算座標↔物理座標の変換の式を定義します。
//それを用いてコンストラクタで各場所での値を事前計算してテーブルに格納します。（table.set_value(honestly_translate))
//シミュレーション中はテーブルを参照します。
//こちらも計算軸クラスと同様に通し番号を設定します。
class Physic_x_
{
private:
public:
    static Value honestly_translate(int calc_x_,int calc_vr,int calc_vt){
        return x_(calc_x_);
    }
    Value translate(int calc_x_,int calc_vr,int calc_vt){
        return x_(calc_x_);
    }
    static const int label = 0;
};

class Physic_vx
{
public:
    static Value honestly_translate(int calc_x_,int calc_vx,int calc_vy){
        // v_x = vr * cos(vt)
        return vx(calc_vx);
    }

    Physic_vx(){}
    Value translate(int calc_x_,int calc_vx,int calc_vy)const{
        return vx(calc_vx);    
    }
    static const int label = 1;
};

class Physic_vy
{
private:
public:
    static Value honestly_translate(int calc_x_,int calc_vx,int calc_vy){
        // v_y = vr * sin(vt) 
        return vy(calc_vy);
    }
    Physic_vy(){}
    Value translate(int calc_x_,int calc_vx,int calc_vy)const{
        return vy(calc_vy);    
    }
    static const int label = 2;
};



//グローバル変数としてインスタンス化しておく。
namespace Global{
    Physic_x_ physic_x_;
    Physic_vx physic_vx;
    Physic_vy physic_vy;
}
/***********************************************
 * 計算軸を物理軸で微分した値の関数を書きます　(始)*
 ***********************************************/


class X__diff_x_
{
public:
    X__diff_x_(){}
    Value at(int calc_x_,int calc_vr,int calc_vt)const{
        return 1./(double)grid_size_x_;
    }
};

class Vx_diff_vx
{
public:
    Vx_diff_vx(){}
    Value at(int calc_x_,int calc_vx,int calc_vy)const{
        return 1./(double)grid_size_vx;
    }
};

class Vy_diff_vy
{
public:
    Vy_diff_vy(){}
    Value at(int calc_x_,int calc_vx,int calc_vy)const{
        return 1./(double)grid_size_vy;
    }
};

//グローバル変数としてインスタンス化
namespace Global{
    const Independent independent;
    const X__diff_x_ x__diff_x_;
    const Vx_diff_vx vx_diff_vx;
    const Vy_diff_vy vy_diff_vy;
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
    x__diff_x_ , independent, independent,
    independent, vx_diff_vx , independent,  
    independent, independent, vy_diff_vy 
);
}
/*******************************************
 * 解くべき移流方程式を定義します。           *
 * df/dt + q/m(v*B)・∇_v f = 0 *
 * を例に定義の仕方を解説                    *
 *******************************************/

//移流項の定義
class Fx_ {
public:
    Fx_(){}
    Value at(int calc_x_,int calc_vx,int calc_vy) const {
        return Global::physic_vx.translate(calc_x_,calc_vx,calc_vy);
    }
};
//------------------------------------------
// 1. q/m (E + v×B)_x
//------------------------------------------
class Fvx {
public:
    Fvx(){}
    Value at(int calc_x_,int calc_vx, int calc_vy) const {
        const Value vy = Global::physic_vy.translate(calc_x_,calc_vx, calc_vy);
        return Q/m*vy*Global::m_field.z + Global::e_field.x;
    }
};

//------------------------------------------
// 2. q/m (E + v×B)_x
//------------------------------------------
class Fvy {
public:
    Fvy(){}
    Value at(int calc_x_,int calc_vx, int calc_vy) const {
        const Value vx = Global::physic_vx.translate(calc_x_,calc_vx, calc_vy);
        return Q/m*(-vx*Global::m_field.z + Global::e_field.y);
    }
};
namespace Global{
    Fx_ flux_x_;
    Fvx flux_vx;
    Fvy flux_vy;
}

/****************************************************************************
 * 次に、Flux計算機を選択します。今回は、Umeda2008を用います。
 ****************************************************************************/

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

//xは周期境界条件
class BoundaryCondition_x_
{
public:
    static const int label = 0;
    template<typename Func>
    static Value left(Func func,int calc_x_,int calc_vx, int calc_vy){
        return func(Axis_x_::num_grid + calc_x_, calc_vx, calc_vy);
    }
    template<typename Func>
    static Value right(Func func,int calc_x_,int calc_vx, int calc_vy){
        return func(calc_x_ - Axis_x_::num_grid, calc_vx, calc_vy);
    }
};


class BoundaryCondition_vx
{
public:
    static const int label = 1;
    //物理的におかしいけど、とりあえずこうしておく
    template<typename Func>
    static Value left(Func func,int calc_x_,int calc_vx, int calc_vy){
        return 0.;
    }

    //物理的におかしいけど、とりあえずこうしておく
    template<typename Func>
    static Value right(Func func,int calc_x_,int calc_vx, int calc_vy){
        return 0.;
    }
};

//thetaは周期境界条件
class BoundaryCondition_vy
{
public:
    static const int label = 2;
    template<typename Func>
    static Value left(Func func,int calc_x_,int calc_vx, int calc_vy){
        return 0.;
    }
    template<typename Func>
    static Value right(Func func,int calc_x_,int calc_vx, int calc_vy){
        return 0.;
    }
};


/*--------------------------------------
 * Pack を用いて境界条件をまとめます。
 *----------------------------------------------*/
namespace Global{
    BoundaryCondition_x_ boundary_condition_x_;
    BoundaryCondition_vx boundary_condition_vx;
    BoundaryCondition_vy boundary_condition_vy;
    
    Pack boundary_condition(
        boundary_condition_x_,
        boundary_condition_vx,
        boundary_condition_vy
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
    Pack operators(physic_x_,physic_vx,physic_vy);
    Pack advections(flux_x_,flux_vx,flux_vy);
    AdvectionEquation equation(dist_function,operators,advections,jacobian,scheme,boundary_condition);
}
Value gauss(Value x,Value y){
    Value sigma = 1.;
    Value m_x = 40.;
    Value m_y =  0.;
    return std::exp(-((x-m_x)*(x-m_x)+(y-m_y)*(y-m_y))/sigma);
}

//可視化のために速度空間で積分する
namespace Global{
    NdTensorWithGhostCell<Value,Axis_x_> f_integrated_by_v;
}

void integrate(){
    for(int i=0;i<Axis_x_::num_grid;i++){
        Global::f_integrated_by_v.at(i)=0.;
        for(int j=0;j<Axis_vx::num_grid;j++){
            for(int k=0;k<Axis_vy::num_grid;k++){
                Global::f_integrated_by_v.at(i)+=Global::dist_function.at(i,j,k);
            }
        }
    }
}

int main(){
    Value dt = 0.01;
    int num_steps = 3000;
    Global::dist_function.set_value(
        [](int calc_x_,int calc_vx,int calc_vy){
            Value x_ = Global::physic_x_.translate(calc_x_,calc_vx,calc_vy);
            Value vx = Global::physic_vx.translate(calc_x_,calc_vx,calc_vy);
            Value vy = Global::physic_vy.translate(calc_x_,calc_vx,calc_vy);
            if(x_<30. && x_>20.)return gauss(vx,vy);
            else{return 0.;}
        }
    );

    Global::m_field.z=m/Q/1.;
    Global::e_field.y=12.*Global::m_field.z;
    Global::e_field.x=0.;
    Global::e_field.z=0.;

    for(int i=0;i<num_steps;i++){
        integrate();
        if(i%5==0)Global::f_integrated_by_v.save_physical("../data/1D2V_ExB_drift_cartesian/"+std::to_string(i/5)+".bin");
        if(i%100==0)std::cout<<i<<std::endl;
        Global::equation.solve<Axis_x_>(dt/2.);
        Global::boundary_manager.apply<Axis_x_>();
        Global::equation.solve<Axis_vx>(dt/2.);
        Global::boundary_manager.apply<Axis_vx>();
        Global::equation.solve<Axis_vy>(dt);
        Global::boundary_manager.apply<Axis_vy>();
        Global::equation.solve<Axis_vx>(dt/2.);
        Global::boundary_manager.apply<Axis_vx>();
        Global::equation.solve<Axis_x_>(dt/2.);
        Global::boundary_manager.apply<Axis_x_>();
    }
    return 0;
}