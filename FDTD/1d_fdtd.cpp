#include "fdtd_solver_1d.h"
#include "../../include.h"
#include <cmath>
using Value = double;

using Axis_x = Axis<0,100,3,3>;

using ElectricField = NdTensorWithGhostCell<Vec3<Value>,Axis_x>;
//E(i,j)=E(x=Δx(i+1/2),t=Δt j)

template<typename Element>
class Pair{
    public:
    using Element_t = Element;
    Element m_half;
    Element p_half;
};

using MagneticField = Pair<NdTensorWithGhostCell<Vec3<Value>,Axis_x>>;
//B(i,j).p_half=B(x=Δx i,t=Δt(j+1/2))

namespace Global{
    ElectricField e_field;
    MagneticField m_field;
}

namespace Global{
    FDTD_solver_1d fdtd_solver(e_field,m_field);
}

void boundary_manager(){
    Global::e_field.at(-1)=Global::e_field.at(Axis_x::num_grid-1);
    Global::e_field.at(Axis_x::num_grid)=Global::e_field.at(0);
    Global::m_field.p_half.at(-1)=Global::m_field.p_half.at(Axis_x::num_grid-1);
    Global::m_field.p_half.at(Axis_x::num_grid)=Global::m_field.p_half.at(0);
    Global::m_field.m_half.at(-1)=Global::m_field.m_half.at(Axis_x::num_grid-1);
    Global::m_field.m_half.at(Axis_x::num_grid)=Global::m_field.m_half.at(0);
}

int main(){

    for(int i=0;i<100;++i){
        Value k = 4./100.;
        Global::e_field.at(i).x=Parameters::c* std::sin(2. * M_PI * k * (Value(i)+0.5));
        Global::e_field.at(i).y=Parameters::c* std::cos(2. * M_PI * k * (Value(i)+0.5));
        Global::m_field.p_half.at(i).x=std::sin(2. * M_PI * k * Value(i) + M_PI/2.);
        Global::m_field.p_half.at(i).y=std::cos(2. * M_PI * k * Value(i) + M_PI/2.);
    }
    double dt = 0.1;
    double courant_v = 0.3;
    double dz = Parameters::c*dt/courant_v;
    Timer timer;
    timer.start();
    for(int t=0;t<10000;++t){
        boundary_manager();
        Global::fdtd_solver.develop(dt/dz);
        if(t%10==0){
            Global::e_field.save_physical("out/e"+std::to_string(t/10)+".bin");
            Global::m_field.p_half.save_physical("out/m"+std::to_string(t/10)+".bin");
        }
    }
    timer.stop();
    std::cout<<timer<<"\n";

    return 0;
}
