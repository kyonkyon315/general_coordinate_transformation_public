#ifndef BOUNDARY_MANAGER_H
#define BOUNDARY_MANAGER_H
#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <optional>
#include <iostream>

#include "block_id2rank.h"
//#include "../utils/arg_changer.h"

#include "axis.h"
#include "BC2block_id.h"
#include "n_d_tensor_with_ghost_cell.h"

#include "comm_graph_constructor.h"
#include "comm_path_generator.h"
#include "node.h"

using Value = double;


//Linear の端における情報の持ち方がわからん。

class SendInfo {
public:
    int dst_rank;
    bool is_my_left;
    bool is_dst_left;
};

inline std::ostream& operator<<(std::ostream& input,const SendInfo& r){
    input<<"dst_rank: "<<r.dst_rank;
    input<<" my_: "<<(r.is_my_left? "left":"right");
    input<<" dst: "<<(r.is_dst_left? "left":"right");
    return input;
}

class RecvInfo {
public:
    int src_rank;
    bool is_src_left;
    bool is_my_left;
};

inline std::ostream& operator<<(std::ostream& input,const RecvInfo& r) {
    input<<"src_rank: "<<r.src_rank;
    input<<" src: "<<(r.is_src_left? "left":"right");
    input<<" my_: "<<(r.is_my_left? "left":"right");
    return input;
}

class CommInfo {
public:

    std::pair<std::optional<SendInfo>,std::optional<RecvInfo>> forward;//時計回り
    std::pair<std::optional<SendInfo>,std::optional<RecvInfo>> backward;//反時計回り

    /*通信手順
    [1]forward.first.has_value && forward.second.has_value　のとき

    普通にsend_and_recv_bytes

    [2]!forward.first.has_value && forward.second.has_value　のとき
    send 側で自分自身とつながっている。これに関してなにもしない
    second 側はrecv_bytes

    [3]forward.first.has_value && !forward.second.has_value　のとき
    recv 側で自分自身とつながっている。自分だけで完結するゴースト処理をこの段階で完了する
    first 側はsend_bytes

    [3]!forward.first.has_value && !forward.second.has_value　のとき
    この場合、軸をブロックで分けていないことになる。
    send 側のゴースト処理を完了する。
    recv 側はなにもしない。
    */

    bool is_edge_forward()const{
        assert(forward.first.has_value()==backward.first.has_value());
        return !forward.first.has_value();
    }

    bool is_edge_backward()const{
        assert(forward.second.has_value()==backward.second.has_value());
        return !forward.second.has_value();
    }
};

inline std::ostream& operator<<(std::ostream& input,CommInfo& r){
    input<<"forward.send_info:\n    ";
    if(r.forward.first.has_value()) input<<r.forward.first.value()<<"\n";
    else input<<"None\n";
    input<<"forward.recv_info:\n    ";
    if(r.forward.second.has_value()) input<<r.forward.second.value()<<"\n";
    else input<<"None\n";
    input<<"backward.send_info:\n    ";
    if(r.backward.first.has_value()) input<<r.backward.first.value()<<"\n";
    else input<<"None\n";
    input<<"backward.recv_info:\n    ";
    if(r.backward.second.has_value()) input<<r.backward.second.value()<<"\n";
    else input<<"None\n";
    return input;
}



//スパコン用のboundary manager では、境界条件の交換と共に、スレッド間での通信も担当する。

template<typename TargetFunc,typename BoundaryCondition, Axis_T... Axes>
class BoundaryManager{
private:
    using TargetFuncValue = typename TargetFunc::Value;
    const BC2BlockId<BoundaryCondition, Axes...> bc2block_id;
    std::tuple<Axes...> axes_tuple;

    const BoundaryCondition& boundary_condition;
    TargetFunc& target_func;
    static constexpr int Dim = TargetFunc::shape.size();
    static constexpr int BC_Dim = BoundaryCondition::get_num_objects();

    static_assert(Dim == BC_Dim,"TaretFuncの次元数とBoundaryConditionの数が一致しません。");
    static_assert(Dim <= sizeof...(Axes));

    BlockId2Rank<Axes...> blockid2rank;
    const int my_world_rank;
    const int thread_num;
    std::array<CommInfo, Dim> comm_info_tichets;

    

    //Axesの入力方法が正しいか確認(開始)
    template<int I = 0>
    static constexpr bool axis_order_checker(){
        if constexpr (I == sizeof...(Axes)) {
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

    template<int I=0>
    void init_comm(){
        if constexpr(I==Dim){
            return;
        }
        else{
            using TargetAxis = std::tuple_element_t<I, std::tuple<Axes...>>;
            CommPathGenerator<BoundaryCondition, Axes...> gen(thread_num);
            std::vector<Node> comm_paths = gen.template get_comm_path<TargetAxis>();
            std::pair<std::vector<Ring>, std::vector<Linear>> ring_and_linear = buildRingsAndLinears(comm_paths);
            auto rings = ring_and_linear.first;
            auto linears = ring_and_linear.second;

            int first_rank_to_comm;
            int second_rank_to_comm;
            for(int i=0;i<rings.size();i++){
                auto& nodes = rings[i].nodes;
                for(int j=0;j<nodes.size();j++){
                    if(nodes[j]==my_world_rank){    
                        first_rank_to_comm = nodes[(j+1)%nodes.size()];//時計回り
                        second_rank_to_comm = nodes[(nodes.size()+j-1)%nodes.size()];//反時計回り
                    }
                }
            }
            for(int i=0;i<linears.size();i++){
                auto& nodes = linears[i].nodes;
                for(int j=0;j<nodes.size();j++){
                    if(nodes[j]==my_world_rank){    
                        first_rank_to_comm = nodes[std::min(j+1,(int)nodes.size()-1)];//右向き
                        second_rank_to_comm = nodes[std::max(j-1,0)];//左向き
                    }
                }
            }
            auto my_node = comm_paths[my_world_rank];
            CommInfo comm_info;
            if(first_rank_to_comm == my_world_rank){
                comm_info.forward.first.reset();
                comm_info.backward.second.reset();
            }
            else{
                auto dst_node = comm_paths[first_rank_to_comm];
                
                std::optional<std::pair<Hand, Hand>> 
                hands_forward = get_connection_hands(comm_paths, my_world_rank, first_rank_to_comm);
                assert(hands_forward.has_value());
                comm_info.forward.first
                    =SendInfo{
                        first_rank_to_comm,
                        hands_forward.value().first == Hand::LEFT,
                        hands_forward.value().second == Hand::LEFT};

                std::optional<std::pair<Hand, Hand>> 
                hands_backward = get_connection_hands(comm_paths, first_rank_to_comm, my_world_rank);
                comm_info.backward.second
                    = RecvInfo{
                        first_rank_to_comm,
                        hands_backward.value().first == Hand::LEFT,
                        hands_backward.value().second == Hand::LEFT
                    };

            }
            if(second_rank_to_comm == my_world_rank){
                comm_info.backward.first.reset();
                comm_info.forward.second.reset();
            }
            else{
                auto dst_node = comm_paths[first_rank_to_comm];
                
                std::optional<std::pair<Hand, Hand>> 
                hands_backward = get_connection_hands(comm_paths, my_world_rank, second_rank_to_comm);
                assert(hands_backward.has_value());
                comm_info.backward.first
                    =SendInfo{
                        second_rank_to_comm,
                        hands_backward.value().first == Hand::LEFT,
                        hands_backward.value().second == Hand::LEFT};

                std::optional<std::pair<Hand, Hand>> 
                hands_forward = get_connection_hands(comm_paths, second_rank_to_comm, my_world_rank);
                comm_info.forward.second
                    = RecvInfo{
                        second_rank_to_comm,
                        hands_forward.value().first == Hand::LEFT,
                        hands_forward.value().second == Hand::LEFT
                    };

            }
            comm_info_tichets[I]=comm_info;

            init_comm<I+1>();
        }
    }

    template<typename TargetAxis>
    void send_and_recv_ghosts(int dst_rank,int src_rank,bool is_my_left,bool is_src_left){
        if(is_my_left && is_src_left)target_func.template send_and_recv_ghosts<TargetAxis,true,true>(dst_rank,src_rank);
        if(!is_my_left && is_src_left)target_func.template send_and_recv_ghosts<TargetAxis,false,true>(dst_rank,src_rank);
        if(is_my_left && !is_src_left)target_func.template send_and_recv_ghosts<TargetAxis,true,false>(dst_rank,src_rank);
        if(!is_my_left && !is_src_left)target_func.template send_and_recv_ghosts<TargetAxis,false,false>(dst_rank,src_rank);
    }

    template<int I>
    using BoundaryCondition_element = typename BoundaryCondition::template element<I>;

    template<Axis_T TargetAxis,bool Is_src_left,bool Is_my_left>
    void ghost_mapping_comm_inner(){
        static_assert(Is_my_left!=Is_src_left);

        constexpr int target_dim = TargetAxis::label; 
        [&]<std::size_t... Idx>(std::index_sequence<Idx...>){
            using TargetSlice = std::conditional_t<
                                    Is_my_left,
                                    Slice<- TargetAxis::L_ghost_length,0>,
                                    Slice<TargetAxis::num_grid,TargetAxis::num_grid+TargetAxis::R_ghost_length>
                                >;
            
            target_func.template set_value_sliced<std::conditional_t<Idx == target_dim, TargetSlice, FullSlice>...>(
                [&]<typename ...Ints>(Ints ...indices)-> TargetFuncValue {
                    return target_func.template buf_at<TargetAxis,Is_src_left>(
                        (Idx == target_dim ? 
                            indices 
                            + (Is_my_left?
                                TargetAxis::num_grid:
                                -TargetAxis::num_grid) : 
                            indices)...
                    );
                }
            );
        }(std::make_index_sequence<Dim>{});
    }

    template<Axis_T TargetAxis,bool Is_src_left,bool Is_my_left>
    void ghost_mapping_comm_edge(int src_rank){
        constexpr int target_dim = TargetAxis::label; 
        [&]<std::size_t... Idx>(std::index_sequence<Idx...>){
            using TargetSlice = std::conditional_t<
                                    Is_my_left,
                                    Slice<- TargetAxis::L_ghost_length,0>,
                                    Slice<TargetAxis::num_grid,TargetAxis::num_grid+TargetAxis::R_ghost_length>
                                >;
            std::tuple<Axes...> src_axes_tuple = axis_instantiator<Axes...>(src_rank);
            
            target_func.template set_value_sliced<std::conditional_t<Idx == target_dim, TargetSlice, FullSlice>...>(
                [&]<typename ...Ints>(Ints ...indices)->TargetFuncValue {
                    if constexpr(Is_my_left){

                        auto idx_tuple = std::tuple{indices...};

                        return [&]<std::size_t... I>(std::index_sequence<I...>) {

                            // ← ここで内側の引数を一度作る
                            auto make_args = [&](auto dim_const) {
                                return [&]<std::size_t... J>(std::index_sequence<J...>) {
                                    return BoundaryCondition_element<target_dim>
                                        ::template left<dim_const>(
                                            (std::get<J>(idx_tuple)
                                            + std::get<J>(axes_tuple).L_id)...);
                                }(std::make_index_sequence<sizeof...(Ints)>{});
                            };

                            return target_func.template buf_at<TargetAxis,Is_src_left>(
                                (
                                    make_args(std::integral_constant<std::size_t, I>{})
                                    - std::get<I>(src_axes_tuple).L_id
                                )...
                            );

                        }(std::make_index_sequence<sizeof...(Ints)>{});
                    }
                    else{
                        auto idx_tuple = std::tuple{indices...};

                        return [&]<std::size_t... I>(std::index_sequence<I...>) {

                            // ← ここで内側の引数を一度作る
                            auto make_args = [&](auto dim_const) {
                                return [&]<std::size_t... J>(std::index_sequence<J...>) {
                                    return BoundaryCondition_element<target_dim>
                                        ::template right<dim_const>(
                                            (std::get<J>(idx_tuple)
                                            + std::get<J>(axes_tuple).L_id)...);
                                }(std::make_index_sequence<sizeof...(Ints)>{});
                            };

                            return target_func.template buf_at<TargetAxis,Is_src_left>(
                                (
                                    make_args(std::integral_constant<std::size_t, I>{})
                                    - std::get<I>(src_axes_tuple).L_id
                                )...
                            );

                        }(std::make_index_sequence<sizeof...(Ints)>{});
                    }
                }
            );
        }(std::make_index_sequence<Dim>{});
    }

    template<Axis_T TargetAxis,bool Is_my_left>
    void ghost_mapping_self(){
        constexpr int target_dim = TargetAxis::label; 
        [&]<std::size_t... Idx>(std::index_sequence<Idx...>){
            using TargetSlice = std::conditional_t<
                                    Is_my_left,
                                    Slice<- TargetAxis::L_ghost_length,0>,
                                    Slice<TargetAxis::num_grid,TargetAxis::num_grid+TargetAxis::R_ghost_length>
                                >;
            
            target_func.template set_value_sliced<std::conditional_t<Idx == target_dim, TargetSlice, FullSlice>...>(
                [&]<typename ...Ints>(Ints ...indices)->TargetFuncValue {
                    if constexpr(Is_my_left){

                        auto idx_tuple = std::tuple{indices...};

                        return [&]<std::size_t... I>(std::index_sequence<I...>) {

                            // ← ここで内側の引数を一度作る
                            auto make_args = [&](auto dim_const) {
                                return [&]<std::size_t... J>(std::index_sequence<J...>) {
                                    return BoundaryCondition_element<target_dim>
                                        ::template left<dim_const>(
                                            (std::get<J>(idx_tuple)
                                            + std::get<J>(axes_tuple).L_id)...);
                                }(std::make_index_sequence<sizeof...(Ints)>{});
                            };

                            return target_func.at(
                                (
                                    make_args(std::integral_constant<std::size_t, I>{})
                                    - std::get<I>(axes_tuple).L_id
                                )...
                            );

                        }(std::make_index_sequence<sizeof...(Ints)>{});
                    }
                    else{

                        auto idx_tuple = std::tuple{indices...};

                        return [&]<std::size_t... I>(std::index_sequence<I...>) {

                            // ← ここで内側の引数を一度作る
                            auto make_args = [&](auto dim_const) {
                                return [&]<std::size_t... J>(std::index_sequence<J...>) {
                                    return BoundaryCondition_element<target_dim>
                                        ::template right<dim_const>(
                                            (std::get<J>(idx_tuple)
                                            + std::get<J>(axes_tuple).L_id)...);
                                }(std::make_index_sequence<sizeof...(Ints)>{});
                            };

                            return target_func.at(
                                (
                                    make_args(std::integral_constant<std::size_t, I>{})
                                    - std::get<I>(axes_tuple).L_id
                                )...
                            );

                        }(std::make_index_sequence<sizeof...(Ints)>{});
                    }
                }
            );
        }(std::make_index_sequence<Dim>{});
    }

    template<Axis_T TargetAxis>
    void apply_helper(const bool is_forward){
        constexpr int target_dim = TargetAxis::label;
        constexpr int num_ghost_grid = TargetAxis::L_ghost_length;//=R_ghost_length
        const int block_id  = std::get<target_dim>(axes_tuple).block_id;

        const CommInfo comm_info = comm_info_tichets[target_dim];
        std::optional<SendInfo> send_info_opt;
        std::optional<RecvInfo> recv_info_opt;
        if(is_forward){
            send_info_opt=comm_info.forward.first;
            recv_info_opt=comm_info.forward.second;
        }
        else{
            send_info_opt=comm_info.backward.first;
            recv_info_opt=comm_info.backward.second;
        }

        //通信
        if(send_info_opt.has_value()&&recv_info_opt.has_value()){
            auto send_info = send_info_opt.value();
            auto recv_info = recv_info_opt.value();
            auto[dst_rank,is_my_left,is_dst_left] = send_info;
            auto[src_rank,is_src_left,is_my_left_] = recv_info;
            assert(is_my_left!=is_my_left_);
            this->template send_and_recv_ghosts<TargetAxis>(dst_rank,src_rank,is_my_left,is_src_left);
        }
        else if(send_info_opt.has_value()&&!recv_info_opt.has_value()){
            auto send_info = send_info_opt.value();
            auto[dst_rank,is_my_left,is_dst_left] = send_info;
            this->template send_and_recv_ghosts<TargetAxis>(dst_rank,-1,is_my_left,false);
        }
        else if(!send_info_opt.has_value()&&recv_info_opt.has_value()){
            auto recv_info = recv_info_opt.value();
            auto[src_rank,is_src_left,is_my_left_] = recv_info;
            this->template send_and_recv_ghosts<TargetAxis>(-1,src_rank,false,is_src_left);
        }
        else{}//なにもしない

        //通信した内容をゴーストセルに埋め込む
        if(recv_info_opt.has_value()){
            if(recv_info_opt.value().is_my_left && block_id >= 1){
                //物理的には内部
                ghost_mapping_comm_inner<TargetAxis, false, true>();
            }
            else if(!recv_info_opt.value().is_my_left && block_id < TargetAxis::num_blocks-1){
                //物理的には内部
                ghost_mapping_comm_inner<TargetAxis, true, false>();
            }
            else{
                auto recv_info = recv_info_opt.value();
                auto[src_rank, is_src_left,is_my_left] = recv_info;

                if(is_src_left && is_my_left) ghost_mapping_comm_edge<TargetAxis, true, true>(src_rank);
                if(is_src_left && !is_my_left) ghost_mapping_comm_edge<TargetAxis, true, false>(src_rank);
                if(!is_src_left && is_my_left) ghost_mapping_comm_edge<TargetAxis, false, true>(src_rank);
                if(!is_src_left && !is_my_left) ghost_mapping_comm_edge<TargetAxis, false, false>(src_rank);
            }
        }
        //自分自身のランクで完結するゴーストセルの更新を行う。
        else{
            if(send_info_opt.has_value()){
                auto send_info = send_info_opt.value();
                bool is_target_left = !send_info.is_my_left;//受け取る側の左右はsend_info.is_my_leftの逆
                if(is_target_left){
                    ghost_mapping_self<TargetAxis, true>();}
                else{
                    ghost_mapping_self<TargetAxis, false>();}
            }
            else{//ブロック分割していない場合
                //forward の時右側
                //backward の時左側
                if(is_forward)
                    ghost_mapping_self<TargetAxis, true>();
                else
                    ghost_mapping_self<TargetAxis, false>();
            }
        }
    }

public:
    BoundaryManager(int my_world_rank,int thread_num,TargetFunc& target_func,const BoundaryCondition& boundary_condition,const Axes&... axes):
        boundary_condition(boundary_condition),
        target_func(target_func),
        axes_tuple(std::tuple<Axes...>{axes...}),
        bc2block_id(my_world_rank,thread_num),
        my_world_rank(my_world_rank),
        thread_num(thread_num)
    {
        init_comm();
        for(int i=0;i<comm_info_tichets.size();i++){
            //std::cout<<"Axis["<<i<<"]:\n"<<comm_info_tichets[i]<<"\n";
        }
    }
    
    template<typename TargetAxis>
    void apply(){
        apply_helper<TargetAxis>(true);//時計回り・右方向
        apply_helper<TargetAxis>(false);//反時計回り・左方向
    }
};
#endif //BOUNDARY_MANAGER_H