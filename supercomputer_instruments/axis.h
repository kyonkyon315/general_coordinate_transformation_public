#ifndef AXIS_H
#define AXIS_H

#include <concepts>

// =====================
// Axis concept
// =====================
template<class A>
concept Axis_T =
requires(const A& a) {
    // compile-time properties
    { A::label } -> std::convertible_to<int>;
    { A::num_grid } -> std::convertible_to<int>;
    { A::num_blocks } -> std::convertible_to<int>;
    { A::num_global_grid } -> std::convertible_to<int>;
    { A::L_ghost_length } -> std::convertible_to<int>;
    { A::R_ghost_length } -> std::convertible_to<int>;

    // runtime properties
    { a.block_id } -> std::convertible_to<int>;
    { a.L_id } -> std::convertible_to<int>;
    { a.R_id } -> std::convertible_to<int>;
};

template<int LABEL,int LENGTH,int NUM_BLOCKS,int LEN_GHOST=3>
class Axis {
    public:
    static constexpr int label = LABEL;
    static constexpr int num_grid = LENGTH;
    static constexpr int L_ghost_length = LEN_GHOST;
    static constexpr int R_ghost_length = LEN_GHOST;
    static constexpr int num_blocks = NUM_BLOCKS;
    static constexpr int num_global_grid = LENGTH * NUM_BLOCKS;
    const int L_id;
    const int R_id;
    const int block_id;
    explicit Axis(int block_id):
        L_id(block_id*num_grid),
        R_id((block_id+1)*num_grid),
        block_id(block_id)
    {}
};

#endif //AXIS_H
