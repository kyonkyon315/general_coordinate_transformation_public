#ifndef BC_2_BLOCK_ID_H
#define BC_2_BLOCK_ID_H


#include "block_id2rank.h"
#include "axis.h"
#include "axis_instantiator.h"
#include <tuple>


template<typename BoundaryCondition,Axis_T... Axes>
class BC2BlockId{
private:
    template<int I>
    using BoundaryElement = typename BoundaryCondition::template element<I>;

    BlockId2Rank<Axes...> block_id2rank;
    const int my_world_rank;
    const int thread_num;
    static constexpr int Dim = sizeof...(Axes);

    //Axesの入力方法が正しいか確認(開始)
    template<int I = 0>
    static constexpr bool axis_order_checker(){
        if constexpr (I == Dim) {
            return true;
        } else {
            return std::tuple_element_t<I, std::tuple<Axes...>>::label == I
                && axis_order_checker<I+1>();
        }
    }
    static_assert(
        axis_order_checker(),
        "Axes... はAxes::labelが小さい順になるように並べてください。\n"
    );
    //Axesの入力方法が正しいか確認(終了)
public:
    BC2BlockId(int my_world_rank,int thread_num):
        my_world_rank(my_world_rank),
        thread_num(thread_num)
    {}
private:
    template<bool Is_tgt_left,int TargetDim>
    std::array<int, Dim> make_ghost_indices(const Axes&... axes)const{
        if constexpr(Is_tgt_left){
            return std::array<int, Dim>{
                (Axes::label == TargetDim
                    ? axes.L_id - 1   // ghost
                    : axes.L_id      // interior
                )...
            };
        }
        else{
            return std::array<int, Dim>{
                (Axes::label == TargetDim
                    ? axes.R_id      // ghost
                    : axes.L_id      // interior
                )...
            };
        }
    }

    template<bool Is_tgt_left,int TargetDim>
    std::tuple<int,bool> calc_candidate(const Axes&... axes)const{
        std::tuple<Axes...> axis_tuple(axes...);
        const std::array<int, Dim> idx = make_ghost_indices<Is_tgt_left,TargetDim>(axes...);
        if constexpr(Is_tgt_left){
            std::array<int,Dim> block_indices={BoundaryElement<TargetDim>::template left<Axes::label>(idx[Axes::label]...)...};
            return std::make_tuple(
                    /*int*/block_id2rank.get_rank(BoundaryElement<TargetDim>::template left<Axes::label>(idx[Axes::label]...)...),
                    /*bool*/block_indices[TargetDim] < std::get<TargetDim>(axis_tuple).L_id 
                );
        }
        else{
            std::array<int,Dim> block_indices={BoundaryElement<TargetDim>::template right<Axes::label>(idx[Axes::label]...)...};
            return std::make_tuple(
                    /*int*/block_id2rank.get_rank(BoundaryElement<TargetDim>::template right<Axes::label>(idx[Axes::label]...)...),
                    /*bool*/block_indices[TargetDim] < std::get<TargetDim>(axis_tuple).L_id 
                );
        }
    }

public:
    template<bool Is_tgt_left,Axis_T TargetAxis> 
    std::tuple<int,bool> get_rcv_rank(const Axes& ...axes)const{
        constexpr int TargetDim = TargetAxis::label;
        auto [candidate_rank, is_src_left] = calc_candidate<Is_tgt_left,TargetDim>(axes...);
        
        //以下、すべてのセルのidを入力してすべて同じランクを指し示すかどうか確認。いまは省略

        return std::make_tuple(candidate_rank,is_src_left);
    }

    //送り先は destination_rank 
    //その送り先の Is_snd_left 側ゴーストに届けるために 
    // 自分のデータは left か right か が is_my_left
    template<bool Is_dst_left, Axis_T TargetAxis>
    std::tuple<int,bool> get_snd_rank()const//(destination_rank, is_my_left)
    {

        for (int rank = 0; rank < thread_num; ++rank) {

            //if (rank == my_world_rank) continue;

            std::tuple<Axes...> axes = axis_instantiator<Axes...>(rank);

            auto [src_rank, is_my_left] =
                std::apply(
                    [&](const Axes&... ax){
                        return this->template
                            get_rcv_rank<Is_dst_left, TargetAxis>(ax...);
                    },
                    axes
                );

            if (src_rank == my_world_rank) {
                // 相手のゴーストを更新するデータ源が自分
                return std::make_tuple(rank, is_my_left);
            }
        }

        // 見つからない場合
        return std::make_tuple(-1, false);
    }

};

#endif// BC_2_BLOCK_ID_H