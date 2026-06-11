#ifndef TUPLE_HEAD_H
#define TUPLE_HEAD_H
#include <tuple>      // スライス機能のために追加
#include <type_traits> // スライス機能のために追加
namespace Utility{
template<std::size_t... Is, typename Tuple>
auto tuple_head_impl(Tuple&& t, std::index_sequence<Is...>) {
    return std::make_tuple(std::get<Is>(std::forward<Tuple>(t))...);
}

template<std::size_t N, typename Tuple>
auto tuple_head(Tuple&& t) {
    static_assert(
        N <= std::tuple_size_v<std::remove_reference_t<Tuple>>,
        "tuple_head<N>: N is larger than tuple size"
    );
    return tuple_head_impl(
        std::forward<Tuple>(t),
        std::make_index_sequence<N>{}
    );
}
}//namespace Utility
#endif //TUPLE_HEAD_H
/*
//使用例
auto t = std::make_tuple(1, 2, 3, 4);

auto h2 = tuple_head<2>(t);  // (1, 2)
auto h3 = tuple_head<3>(t);  // (1, 2, 3)


*/