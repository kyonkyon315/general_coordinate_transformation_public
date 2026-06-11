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
using Axis_v_ = Axis<1,100,3,3>;

//電子分布関数の型を定義
//先頭に入力する型はテンソルの値の型です。その後に続く軸は、通し番号が小さいものほど左に入力してください。
using DistributionFunction = NdTensorWithGhostCell<Value,Axis_x_,Axis_v_>;

//電場の型を定義
using ElectricField = NdTensorWithGhostCell<Vec3<Value>,Axis_x_>;

//磁場の型を定義
using MagneticField = NdTensorWithGhostCell<Vec3<Value>,Axis_x_>;

//グローバル変数としてインスタンス化しておく。
namespace Global{
    DistributionFunction dist_function;
    ElectricField e_field;
    MagneticField m_field;
}

/***********************************************
 * 物理空間と計算空間の関係を表す関数を書きます(始)*
 ***********************************************/

// --- グローバル定数とヘルパー関数の定義 ---

constexpr Value grid_size_x_ = 5.;
const Value grid_size_v_ = 2.;

Value v(int calc_v_){ return grid_size_v_ * (double)(calc_v_-Axis_v_::num_grid/2);}


// --- 物理量クラス ---
//honestly_translateで計算座標↔物理座標の変換の式を定義します。
//それを用いてコンストラクタで各場所での値を事前計算してテーブルに格納します。（table.set_value(honestly_translate))
//シミュレーション中はテーブルを参照します。
//こちらも計算軸クラスと同様に通し番号を設定します。

class Physic_x_
{
public:
    Physic_x_(){}
    Value translate(int calc_x,int calc_vr)const{
        return grid_size_x_*calc_x;
    }
    static const int label = 0;
};

class Physic_v_
{
public:
    Physic_v_(){}
    Value translate(int calc_x_,int calc_v_)const{
        return v(calc_v_);    
    }
    static const int label = 1;
};


//グローバル変数としてインスタンス化しておく。
namespace Global{
    Physic_x_ physic_x_;
    Physic_v_ physic_v_;
}
/***********************************************
 * 計算軸を物理軸で微分した値の関数を書きます　(始)*
 ***********************************************/



class X__diff_x_
{
public:
    X__diff_x_(){}
    Value at(int calc_x_,int calc_v_)const{
        return 1./grid_size_x_;
    }
};

class V__diff_v_
{
public:
    V__diff_v_(){}
    Value at(int calc_x_,int calc_v_)const{
        return 1./grid_size_v_;
    }
};

//グローバル変数としてインスタンス化
namespace Global{
    const Independent independent;
    const X__diff_x_ x__diff_x_;
    const V__diff_v_ v__diff_v_;
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
    x__diff_x_ , independent, 
    independent, v__diff_v_ 
);
}
/*******************************************
 * 解くべき移流方程式を定義します。           *
 * df/dt + v_x df/dx + q/m(E+v*B)・∇_v f = 0 *
 * を例に定義の仕方を解説                    *
 *******************************************/

//移流項の定義
//------------------------------------------
// 1. v_x * df/dx
//------------------------------------------
class Fx_ {
public:
    Fx_(){}
    Value at(int calc_x_, int calc_vr) const {
        return Global::physic_v_.translate(calc_x_, calc_vr);
    }
};

//------------------------------------------
// 2. q/m (E + v×B)_x
//------------------------------------------
class Fv_ {
private:
public:
    Fv_(){}

    Value at(int calc_x_, int calc_vr) const {
        return Q/m * (      Global::e_field.at(calc_x_).x );
    }
};
namespace Global{
    Fx_ flux_x_;
    Fv_ flux_v_;
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
    static Value left(Func func,int calc_x_, int calc_v_){
        return func(-calc_x_, calc_v_);
    }
    template<typename Func>
    static Value right(Func func,int calc_x_, int calc_v_){
        return func(calc_x_ - Axis_x_::num_grid, calc_v_);
    }
};

class BoundaryCondition_v_
{
public:
    static const int label = 1;
    template<typename Func>
    static Value left(Func func,int calc_x_, int calc_v_){
        return func(calc_x_, -calc_v_);
    }

    //物理的におかしいけど、とりあえずこうしておく
    template<typename Func>
    static Value right(Func func,int calc_x_, int calc_v_){
        return func(calc_x_, calc_v_-Axis_x_::num_grid);
    }
};

/*--------------------------------------
 * Pack を用いて境界条件をまとめます。
 *----------------------------------------------*/
namespace Global{
    BoundaryCondition_x_ boundary_condition_x_;
    BoundaryCondition_v_ boundary_condition_v_;
    
    Pack boundary_condition(
        boundary_condition_x_,
        boundary_condition_v_
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
    Pack operators(physic_x_,physic_v_);
    Pack advections(flux_x_,flux_v_);
    AdvectionEquation equation(dist_function,operators,advections,jacobian,scheme,boundary_condition);
}

int main(){
    Value dt = 0.1;
    int num_steps = 100;
    for(int i=0;i<num_steps;i++){
        std::cout<<i<<std::endl;
        Global::equation.solve<Axis_v_>(dt/2.);
        Global::boundary_manager.apply<Axis_v_>();
        Global::equation.solve<Axis_x_>(dt);
        Global::boundary_manager.apply<Axis_x_>();
        Global::equation.solve<Axis_v_>(dt/2.);
        Global::boundary_manager.apply<Axis_v_>();
    }
    return 0;
}