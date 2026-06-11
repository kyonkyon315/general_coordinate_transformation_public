#ifndef CELL_MAP_MAKER_H
#define CELL_MAP_MAKER_H
#include <tuple>
#include <utility>
#include <vector>
#include "axis_instantiator.h"
#include "axis.h"
#include "n_d_tensor_with_ghost_cell.h"
template <typename Value,typename Operators_tuple,typename... Axes>
class CellMapMaker{
private:
    const std::tuple<Axes...> axes;
    Operators_tuple operators;

    template<int Dim>
    using Operator = std::tuple_element_t<Dim, Operators_tuple>;

    static constexpr int N_dim = sizeof...(Axes);
    static constexpr std::array<Index, N_dim> shape = {Axes::num_grid...};
    static constexpr int N_data = []()constexpr{
        int retval = 1;
        for(int i=0;i<shape.size();i++){
            retval*= shape[i];
        }
        return retval;
    };
    static constexpr int N_vertex = (1ULL<<N_dim);

    using AxisVertex = Axis<N_dim,N_vertex,1,0>;
    NdTensorWithGhostCell<Value, Axes...,AxisVertex> data;

    template<int TargetDim,typename... Ints>
    Value get_cood_val_in_one_vertex(const unsigned long long is_left_info,Ints... indices){
        std::array<int,N_dim> inputs ={indices...};
        for(int i=0;i<N_dim;i++){
            inputs[i] += ((is_left_info&&(1ULL<<i))? 1 : 0);
        }
        return [&]<int... Idx>(std::index_sequence<Idx...>){
            return std::get<TargetDim>(operators).translate((inputs[Idx])...);
        }(std::make_index_sequence<N_dim>{});
    }

    template<int Depth,typename... Ints>
    void calc_helper(Ints... indices){
        if constexpr(Depth==N_dim){
            for(unsigned long long vertex = 0ULL; vertex<N_vertex; ++vertex){
                data.at(indices...,vertex)= get_cood_val_in_one_vertex<int TargetDim>(vertex, indices...)
            }
        }
        else {
            for(int i=0;i<shape[Depth];++i){
                calc_helper<Depth+1>(indices...,i);
            }
        }
    }
    template <std::size_t... I>
    static Operators_tuple make_tuple_impl(int x, std::index_sequence<I...>) {
        return Operators_tuple( (static_cast<void>(I), typename std::tuple_element<I, Operators_tuple>::type(x))... );
    }

    static Operators_tuple make_tuple(int x) {
        constexpr std::size_t N = std::tuple_size<Operators_tuple>::value;
        return make_tuple_impl(x, std::make_index_sequence<N>{});
    }



public:
    CellMapMaker(const int my_world_rank):
        axes(axis_instantiator<Axes...>(my_world_rank)),
        operators(make_tuple(my_world_rank))
    {
        data.resize(N_dim*N_vertex*N_data);

    }

};

#endif //CELL_MAP_MAKER_H
