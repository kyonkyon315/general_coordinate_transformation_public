#ifndef ADVECTION_EQUATION_H
#define ADVECTION_EQUATION_H

#if defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define ALWAYS_INLINE inline
#endif

#include "independent.h"
#include "utils/arg_changer.h"
#include "utils/tuple_head.h"
#include <type_traits>
#include <array>
#include <utility>
#include <omp.h>
#include <cmath>
#include "none.h"

using Value = double;
template<typename TargetFunction,typename Operators,typename Advections,typename Jacobian,typename Scheme,typename BoundaryCondition,typename Current>
class AdvectionEquation
{
private:
    TargetFunction& target_func;
    TargetFunction func_buffer;
    Current& current;
    const Operators& operators;
    const Advections& advections;
    const Jacobian& jacobian;
    const Scheme& scheme;
    const BoundaryCondition& boundary_condition;

    static constexpr int dimension = TargetFunction::get_dimension();
    static constexpr int real_dimension = Current::get_dimension();
    static constexpr int velo_dimension = dimension - real_dimension;

    
    static constexpr bool need_current = []()constexpr{
        if constexpr(std::is_same_v<Current,None_current>)return false;
        return true;
    }();

    static_assert(
        []()constexpr{
            if(!need_current)return true;
            if(velo_dimension < 0)return false;
            for(int i = 0;i < real_dimension; ++i){
                if(TargetFunction::shape[i]!=Current::shape[i])return false;
            }
            return true;
        }()
        ,"電流の shape が分布関数の実空間部の shape と一致しません。"
    );

    static constexpr int L = Scheme::used_id_left;
    static constexpr int R = Scheme::used_id_right;


    
    //連鎖率を用いて、計算空間でのフラックスを計算します。
    template<int I,int Target_Dim,typename... Ints>
    ALWAYS_INLINE
    Value advection_in_calc_space_helper(Ints... indices){
        using E = typename Jacobian::element_t<Target_Dim,I>;
        if constexpr(I==dimension-1){
            if constexpr(std::is_same_v<E, Independent>){
                //こんなことしなくても最適化で０の項はなくなるかも。
                return 0.;
            }
            return jacobian.template get_element<Target_Dim,I>().at(indices...)
                    *advections.template get_object<I>().at(indices...);
        }
        else{
            if constexpr(std::is_same_v<E, Independent>){
                //こんなことしなくても最適化で０の項はなくなるかも。
                return advection_in_calc_space_helper<I+1,Target_Dim>(indices...);
            }
            return jacobian.template get_element<Target_Dim,I>().at(indices...)
                    *advections.template get_object<I>().at(indices...)
                    +advection_in_calc_space_helper<I+1,Target_Dim>(indices...);
        }
    }

    template<int Target_Dim,typename... Ints>
    ALWAYS_INLINE
    Value advection_in_calc_space(Ints... indices){
        return advection_in_calc_space_helper<0,Target_Dim>(indices...);
    }


    template<typename... Indices>
    void sum_current_helper(const Value dt,const Value U_i_p_half,Indices... indices){
            
        static_assert(sizeof...(Indices) >= real_dimension,
                    "not enough indices");

        if constexpr(real_dimension == 1){
            current.at(std::get<0>(std::tie(indices...))).z -= U_i_p_half/dt;
        }
        else{
            throw 1;
            auto idx = std::make_tuple(indices...);
            std::apply(
                [&](auto... head){
                    current.at(head...).z += U_i_p_half/dt;
                    //current_tilde = U/dt_tilde
                },
                Utility::tuple_head<real_dimension>(idx)
            );
        }
    }

    // 再帰ヘルパ：indices を集める
    template<int Depth,int Dim,int Target_Dim,typename... Ints>
    ALWAYS_INLINE
    void solve_helper(Value dt, Ints... indices){
        if constexpr(Depth == Dim){
            Value Um, Up;
            solve_leaf<Target_Dim>(dt, Up, Um, indices...);

            //[TODO]今のところ実空間は一次元しか考慮していない。
            //実空間の移流を求めたときに、電流が計算される。
            if constexpr(Target_Dim == 0 && need_current){ 
                //ここで電流を保存したいね。indices の数をreal_dim の数に減らしたい
                sum_current_helper(dt,Up,indices...);
            }
            //増加分を保存
            func_buffer.at(indices...) = Um-Up;
        }
        else if constexpr(Depth==0){
            //[@TODO]Dim==1のときはomp発動しないようにした方がいいかも。
            constexpr int axis_len = TargetFunction::shape[Depth];
            #pragma omp parallel for
            for(int i=0;i<axis_len;++i){
                if constexpr(Target_Dim ==0 && need_current){
                    //current.at(i).z = 0.;
                }
                // 再帰：indices に i を追加
                solve_helper<Depth+1,Dim,Target_Dim>(dt, indices..., i);
            }
        }
        else{
            constexpr int axis_len = TargetFunction::shape[Depth];
            for(int i=0;i<axis_len;++i){
                // 再帰：indices に i を追加
                solve_helper<Depth+1,Dim,Target_Dim>(dt, indices..., i);
            }
        }
    }

    // leaf: index_sequence を生成して「index-first」ヘルパを呼ぶ
    template<int Target_Dim, typename... Ints>
    ALWAYS_INLINE
    void solve_leaf(Value dt, Value& Up, Value& Um, Ints... indices){
        constexpr std::size_t stencil_size = (R - L + 1);
        solve_leaf_impl_indices<Target_Dim>(dt, Up, Um, std::make_index_sequence<stencil_size>{}, indices...);
    }

    // helper: index_sequence を最初の引数に置く（※これで推論が安定）
    template<int Target_Dim, std::size_t... Is, typename... Ints>
    ALWAYS_INLINE
    void solve_leaf_impl_indices(Value dt, Value& Up, Value& Um, std::index_sequence<Is...>, Ints... indices){
        // コンパイル時に確定するオフセット配列
        static constexpr int stencil_offsets[] = { (int(Is) + L)... };

        // チェーンルールによる advection 計算
        Value advection_p_1 = Utility::arg_changer<Value,Target_Dim,1>(
                            [this](auto... idxs) -> Value{
                                return this->advection_in_calc_space<Target_Dim>(idxs...);
                            },
                            indices...
                        );
        Value advection = Utility::arg_changer<Value,Target_Dim,0>(
                            [this](auto... idxs) -> Value{
                                return this->advection_in_calc_space<Target_Dim>(idxs...);
                            },
                            indices...
                        );
        Value advection_m_1 = Utility::arg_changer<Value,Target_Dim,-1>(
                            [this](auto... idxs) -> Value{
                                return this->advection_in_calc_space<Target_Dim>(idxs...);
                            },
                            indices...
                        );
        Value nyu_p_half = - dt * (advection+advection_p_1)/2.;
        Value nyu_m_half = - dt * (advection+advection_m_1)/2.;
        
        this->scheme.calc_U(
            Utility::arg_changer<Value, Target_Dim, stencil_offsets[Is]>(
                [this](auto... idxs) -> Value {
                    return this->target_func.at(idxs...);
                },
                indices...
            )...,
            nyu_m_half, nyu_p_half, Um, Up
        );

        if(std::isnan(Um)||std::isnan(Up)){
            ((std::cout<<indices<<" "),...);
            std::cout<<"nyu_p_half: "<<Up<<"\n";
            std::cout<<"nyu_m_half: "<<Um<<"\n";
            std::cout<<"\n";
            throw 1;
        }
    }

public:
    AdvectionEquation(
        TargetFunction& target_func,
        const Operators& operators,
        const Advections& advections,
        const Jacobian& jacobian,
        const Scheme& scheme,
        const BoundaryCondition& boundary_condition,
        Current& current
    ):
        target_func(target_func),
        operators(operators),
        advections(advections),
        jacobian(jacobian),
        scheme(scheme),
        boundary_condition(boundary_condition),
        current(current)
    {
        static_assert(target_func.get_dimension()==operators.get_num_objects());
        static_assert(target_func.get_dimension()==advections.get_num_objects());
    }
    template<typename CalcAxis>
    void solve(Value dt){
        constexpr int target_dim = CalcAxis::label;
        // dfを計算してfunc_bufferに格納
        solve_helper<0, dimension, target_dim>(dt);

        //dfをfに足し込む（関数の更新）
        target_func.add(func_buffer);
    }
};
#endif //ADVECTION_EQUATION_H