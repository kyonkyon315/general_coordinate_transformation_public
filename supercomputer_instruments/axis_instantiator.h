#ifndef AXIS_INSTANTIATOR_H
#define AXIS_INSTANTIATOR_H

#include <tuple>
#include <array>
#include <utility>
#include <stdexcept>
using Index = int;

namespace detail {


template <typename... Axes>
constexpr Index total_num_blocks() {
    return (Axes::num_blocks * ...);
}

template <typename... Axes>
constexpr std::array<Index, sizeof...(Axes)>
make_num_blocks_array() {
    return {Axes::num_blocks...};
}

template <typename... Axes, std::size_t... Is>
std::tuple<Axes...>
axis_instantiator_impl(Index rank, std::index_sequence<Is...>) {
    constexpr auto num_blocks = make_num_blocks_array<Axes...>();

    // rank を mixed radix で分解
    std::array<Index, sizeof...(Axes)> block_ids{};
    for (int i = sizeof...(Axes) - 1; i >= 0; --i) {
        block_ids[i] = rank % num_blocks[i];
        rank /= num_blocks[i];
    }

    // pack 展開で tuple を構築
    return std::tuple<Axes...>(Axes(block_ids[Is])...);
}

} // namespace detail

template <typename... Axes>
std::tuple<Axes...> axis_instantiator(Index rank) {
    constexpr Index total_block = detail::total_num_blocks<Axes...>();

    if (rank < 0 || rank >= total_block) {
        throw std::out_of_range(
            "axis_instantiator: rank is out of valid range");
    }

    return detail::axis_instantiator_impl<Axes...>(
        rank, std::index_sequence_for<Axes...>{});
}

#endif // AXIS_INSTANTIATOR_H
