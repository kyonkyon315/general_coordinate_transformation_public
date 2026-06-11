#ifndef FDTD_SOLVER_1D
#define FDTD_SOLVER_1D
#include "../parameters.h"
#include "../normalization.h"
using Value = double;
using Index = int;
//軸の方向はz　一次元直線座標専門のfdtd
template<typename ElectricField,typename MagneticField, typename Current>
class FDTD_solver_1d{
    using MagneticField_t = typename MagneticField::Element_t;
    static_assert(ElectricField::shape.size()==1,"electric field must be 1d.\n");
    static_assert(MagneticField_t::shape.size()==1,"magnetic field must be 1d.\n");
    static_assert(ElectricField::shape[0]==MagneticField_t::shape[0],"sizes of electric field and magnetic field mismatch.\n");

    static constexpr Index num_grid = ElectricField::shape[0];
    static constexpr Value curlB_coef = Norm::Coef::maxwell_curlB_coef;
    static constexpr Value J_coef = Norm::Coef::maxwell_current_coef;

    private:
    ElectricField& e_field;
    MagneticField& m_field;
    Current& current;

    void develop_m(Value dt, Value dz){
        swap(m_field.p_half,m_field.m_half);
        for(Index i=0;i<num_grid;++i){
            m_field.p_half.at(i).x = m_field.m_half.at(i).x 
                + dt/dz*(e_field.at(i).y - e_field.at(i-1).y);
            m_field.p_half.at(i).y = m_field.m_half.at(i).y 
                - dt/dz*(e_field.at(i).x - e_field.at(i-1).x);
        }
    }

    void develop_e(Value dt,Value dz){
        const Value coef = curlB_coef * dt/dz;
        const Value jcoef = J_coef * dt;
        for(Index i=0;i<num_grid;++i){
            e_field.at(i).x -= coef * (m_field.p_half.at(i+1).y - m_field.p_half.at(i).y);
            e_field.at(i).y += coef * (m_field.p_half.at(i+1).x - m_field.p_half.at(i).x);

            e_field.at(i).x -= jcoef * current.at(i).x;
            e_field.at(i).y -= jcoef * current.at(i).y;
            e_field.at(i).z -= jcoef * current.at(i).z;
        }
    }

    public:
    FDTD_solver_1d(ElectricField& e_field,MagneticField& m_field,Current& current):
        e_field(e_field),
        m_field(m_field),
        current(current)
    {}

    void develop(Value dt_tilde, Value dz_tilde){
        develop_e(dt_tilde, dz_tilde);
        develop_m(dt_tilde, dz_tilde);
    }
};

#endif //FDTD_SOLVER_1D