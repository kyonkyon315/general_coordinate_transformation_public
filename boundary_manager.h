#ifndef BOUNDARY_MANAGER_H
#define BOUNDARY_MANAGER_H
#include <iostream>
#include <array>
#include "n_d_tensor_with_ghost_cell.h"
/*
template<typename T,typename... Axes>
class NdTensorWithGhostCell {
public:
    // 軸ごとの長さをコンパイル時に収集
    static constexpr std::array<Index, sizeof...(Axes)> shape = {Axes::num_grid...};
    // 軸ごとのマイナス側ゴーストセル数をコンパイル時に収集
    static constexpr std::array<Index, sizeof...(Axes)> L_ghost_lengths = {Axes::L_ghost_length...};
    // 軸ごとのプラス側ゴーストセル数をコンパイル時に収集
    static constexpr std::array<Index, sizeof...(Axes)> R_ghost_lengths = {Axes::R_ghost_length...};
*/
template<typename TargetFunc,typename BoundaryCondition>
class BoundaryManager{
private:
    const BoundaryCondition& boundary_condition;
    TargetFunc& target_func;
    static constexpr int Dim = TargetFunc::shape.size();

    template<int Target_dim,std::size_t... Is>
    void apply_helper_l(std::index_sequence<Is...>){
        constexpr int ghost_len = TargetFunc::L_ghost_lengths[Target_dim];
        target_func.template set_value_sliced<
            std::conditional_t<Is == Target_dim, Slice<-ghost_len, 0>, FullSlice>...
        >(
            // The callable that set_value_sliced expects: it will be invoked with
            // the current (physical) indices for the tensor. We create a wrapper
            // that calls the appropriate boundary-condition's left(...) method,
            // and we pass a small "inner func" that redirects to target_func.at(...)
            [this](auto... idxs) -> decltype(auto) {
                // inner func to be passed to boundary condition: it should provide
                // access to the original function (target_func.at) for arbitrary indices
                auto inner = [this](auto&&... inner_idxs) -> decltype(auto) {
                    return target_func.at(inner_idxs...);
                };
                // call the element instance's left method
                return this->boundary_condition.template get_object<Target_dim>().left(inner, idxs...);
            }
        );
    }

    template<int Target_dim, std::size_t... Is>
    void apply_helper_r(std::index_sequence<Is...>) {
        constexpr int ghost_len = TargetFunc::R_ghost_lengths[Target_dim];
        constexpr int len = TargetFunc::shape[Target_dim];

        target_func.template set_value_sliced<
            std::conditional_t<Is == Target_dim, Slice<len, len + ghost_len>, FullSlice>...
        >(
            [this](auto... idxs) -> decltype(auto) {
                auto inner = [this](auto&&... inner_idxs) -> decltype(auto) {
                    return target_func.at(inner_idxs...);
                };
                return this->boundary_condition.template get_object<Target_dim>().right(inner, idxs...);
            }
        );
    }

public:
    BoundaryManager(TargetFunc& target_func,const BoundaryCondition& boundary_condition):
        boundary_condition(boundary_condition),
        target_func(target_func)
    {
    }
    //Pack<BoundaryCondition_x_,BoundaryCondition_vr,BoundaryCondition_vt,BoundaryCondition_vp>::element<1>::left(hoge,1,2,3,4);
    
    template<typename TargetAxis>
    void apply(){
        constexpr int target_dim = TargetAxis::label;
        apply_helper_l<target_dim>(std::make_index_sequence<Dim>{});
        apply_helper_r<target_dim>(std::make_index_sequence<Dim>{});
    }
};
#endif //BOUNDARY_MANAGER_H