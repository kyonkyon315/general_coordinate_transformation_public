#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include "parameters.h"
#include "n_d_tensor_with_ghost_cell.h"
//1d専用
using Value = double;

/*
//電場の型を定義
using ElectricField = NdTensorWithGhostCell<Vec3<Value>,Axis_x_>;
//E(i,j)=E(x=Δx(i+1/2),t=Δt j)
*/

template<typename DistFunc,typename ElectricField>
class PoissonSolver{
public:
    static_assert(ElectricField::shape.size()==1,
        "this solver is only for 1d");
    static_assert(ElectricField::shape[0]==DistFunc::shape[0],
        "ElectricField::shape[0] and DistFunc::shape[0] missmatch");
    static constexpr int num_grid = ElectricField::shape[0];
    static constexpr auto shape = DistFunc::shape;
private:
    template<typename DistFunc,int Depth,typename Ints>
    Value solve_helper(const DistFunc& dist_f,Ints ...indices){
        if constexpr(Depth == shape.size()){
            return dist_f.at(indices);
        }
        else{
            Value ans = 0.;
            for(Index i =0;i<shape[Depth];++i){
                ans+=solve_helper<DistFunc,Depth+1,Ints...,Index>(dist_f,indices,i);
            }
            return ans;
        }
    }
public:
    void solve(const DistFunc& dist_f,ElectricField& e_field){
        for(Index i=0;i<num_grid;++i){
            e_field[i].z = solve_helper<DistFunc,1,Index>(i)*Q/epsilon_0;
        }
    }
};

#endif //POISSON_SOLVER_H