#ifndef COMM_PATH_GENERATOR
#define COMM_PATH_GENERATOR

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>
#include "node.h"
#include "block_id2rank.h"
#include "axis_instantiator.h"
#include "tuple"

template<typename BoundaryCondition, typename... Axes>
class CommPathGenerator{
private:
    BlockId2Rank<Axes...> block_id2rank;
    static constexpr int DIM = sizeof...(Axes);
    const int thread_num;
    template<int I>
    using BoundaryElement = typename BoundaryCondition::template element<I>;

    template<bool Is_tgt_left,int TargetDim>
    std::array<int, DIM> make_ghost_indices(const Axes&... axes)const{
        if constexpr(Is_tgt_left){
            return std::array<int, DIM>{
                (Axes::label == TargetDim
                    ? axes.L_id - 1   // ghost
                    : axes.L_id      // interior
                )...
            };
        }
        else{
            return std::array<int, DIM>{
                (Axes::label == TargetDim
                    ? axes.R_id      // ghost
                    : axes.L_id      // interior
                )...
            };
        }
    }

    
    template<bool Is_tgt_left,int TargetDim>
    std::tuple<int,bool> calc_candidate(const Axes&... axes)const{//src_rank, from whether left or right data come from. 
        std::tuple<Axes...> axis_tuple(axes...);
        const std::array<int, DIM> idx = make_ghost_indices<Is_tgt_left,TargetDim>(axes...);
        using TargetAxis = std::tuple_element_t<TargetDim, std::tuple<Axes...>>;

        constexpr int BC_Dim = BoundaryCondition::get_num_objects();
        static_assert(BC_Dim <= DIM,"BoundaryCondition の要素数 <= Axesの数 である必要があります。" );
        //if constexpr(BC_Dim == DIM){
        if constexpr(false){
            if constexpr(Is_tgt_left){
                std::array<int,DIM> src_indices={BoundaryElement<TargetDim>::template left<Axes::label>(idx[Axes::label]...)...};
                
                return std::make_tuple(
                        /*int*/block_id2rank.get_rank((src_indices[Axes::label]/Axes::num_grid)...),
                        /*bool*/src_indices[TargetDim] % TargetAxis::num_grid <  TargetAxis::L_ghost_length//=R_ghost_length
                    );
            }
            else{
                std::array<int,DIM> src_indices={BoundaryElement<TargetDim>::template right<Axes::label>(idx[Axes::label]...)...};
                return std::make_tuple(
                        /*int*/block_id2rank.get_rank((src_indices[Axes::label]/Axes::num_grid)...),
                        /*bool*/src_indices[TargetDim] % TargetAxis::num_grid <  TargetAxis::L_ghost_length//=R_ghost_length
                    );
            }
        }
        //else{
            //BoundaryCondition の要素数＜Axesの数ならば、あまりのAxesは固定して送信先を決める。電場、磁場の境界条件をつくるのに便利。
            static_assert(TargetDim < BC_Dim,"CommPathGenerator::calc_candidate()->  TargetDim < BC_Dim が必要です。");
            std::array<int,DIM> src_indices;
            std::array<int,BC_Dim> a;
            [&]<std::size_t ...I>(std::index_sequence<I...>){
                // ← ここで内側の引数を一度作る
                auto make_args = [&](auto dim_const) {
                    return  [&]<std::size_t... J>(std::index_sequence<J...>) {
                                if constexpr(Is_tgt_left){
                                    return BoundaryElement<TargetDim>
                                        ::template left<dim_const>(
                                            (idx[J])...);
                                }
                                else{
                                    return BoundaryElement<TargetDim>
                                        ::template right<dim_const>(
                                            (idx[J])...);
                                }
                            }(std::make_index_sequence<BC_Dim>{});
                };
                a ={make_args(std::integral_constant<std::size_t, I>{})...};
            }(std::make_index_sequence<BC_Dim>{});

            for(int i=0;i<BC_Dim;i++){
                src_indices[i] = a[i];
            }
            for(int i=BC_Dim;i<DIM;i++){
                src_indices[i] = idx[i];
            }
            return std::make_tuple(
                    /*int*/block_id2rank.get_rank((src_indices[Axes::label]/Axes::num_grid)...),
                    /*bool*/src_indices[TargetDim] % TargetAxis::num_grid <  TargetAxis::L_ghost_length//=R_ghost_length
                );

        //}
    }


public:
    CommPathGenerator(int thread_num):
        thread_num(thread_num)
    {
    }
    template<typename TargetAxis>
    std::vector<Node> get_comm_path()const{
        std::vector<Node> retval(thread_num);
        constexpr int TargetDim = TargetAxis::label;

        //まずは内部の通信パスを登録する。
        //注目している軸について、block_idを±1するだけでよい。
        
        for(int rank=0;rank<thread_num;rank++){
            //std::cout<<rank<<" "<<std::flush;
            std::tuple<Axes...> axes = axis_instantiator<Axes...>(rank);
            auto& target_axis = std::get<TargetDim>(axes);
            if (target_axis.block_id != target_axis.num_blocks-1){
                int dst_rank = [&]<std::size_t... Is>(std::index_sequence<Is...>) -> int 
                    {
                        return block_id2rank.get_rank(
                                    (Is == TargetDim ? 
                                        std::get<Is>(axes).block_id+1 : 
                                        std::get<Is>(axes).block_id
                                    )...);
                    }(std::make_index_sequence<DIM>{});
                    //std::cout<<rank<<"->"<<dst_rank<<" ";
                retval[rank].right = Endpoint{dst_rank, Hand::LEFT};
            }
            if (target_axis.block_id != 0){
                int dst_rank = [&]<std::size_t... Is>(std::index_sequence<Is...>) -> int 
                    {
                        return block_id2rank.get_rank(
                                    (Is == TargetDim ? 
                                        std::get<Is>(axes).block_id-1 : 
                                        std::get<Is>(axes).block_id
                                    )...);
                    }(std::make_index_sequence<DIM>{});
                retval[rank].left = Endpoint{dst_rank, Hand::RIGHT};
            }
        }
        //内部の通信パスの登録　終了
        

        //境界条件のための通信パスの登録
        for(int rank=0;rank<thread_num;rank++){
            std::tuple<Axes...> axes = axis_instantiator<Axes...>(rank);
            auto& target_axis = std::get<TargetDim>(axes);
            if (target_axis.block_id == 0){
                std::tuple<int,bool> dst_rank_and_is_left = [&]<std::size_t... Is>(std::index_sequence<Is...>) -> std::tuple<int,bool> {
                        return this->template calc_candidate<true, TargetDim>((std::get<Is>(axes))...);
                    }(std::make_index_sequence<DIM>{});
                auto& [dst_rank, is_left] = dst_rank_and_is_left;
                retval[rank].left = Endpoint{dst_rank, (is_left ? Hand::LEFT : Hand::RIGHT)};
            }
            if (target_axis.block_id == TargetAxis::num_blocks -1){
                std::tuple<int,bool> dst_rank_and_is_left = [&]<std::size_t... Is>(std::index_sequence<Is...>) -> std::tuple<int,bool>
                    {
                        return this->calc_candidate<false, TargetDim>((std::get<Is>(axes))...);
                    }(std::make_index_sequence<DIM>{});
                auto& [dst_rank, is_left] = dst_rank_and_is_left;
                retval[rank].right = Endpoint{dst_rank, (is_left ? Hand::LEFT : Hand::RIGHT)};
            }
        }
        return retval;
    }
};

#endif //COMM_PATH_GENERATOR