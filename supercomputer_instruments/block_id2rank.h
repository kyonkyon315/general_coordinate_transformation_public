#ifndef BLOCK_ID2RANK_H
#define BLOCK_ID2RANK_H

#include <array>
#include <stdexcept>
#include <iostream>

template<typename... Axes>
class BlockId2Rank {
private:
    static constexpr int ndim = sizeof...(Axes);

    // 各軸の num_blocks を compile-time に保持
    static constexpr std::array<int, ndim> num_blocks = {
        Axes::num_blocks...
    };
public:
    // Axis オブジェクトを受け取る（block_id は使わなくても設計上自然）
    explicit BlockId2Rank(const Axes&...) {}

    explicit BlockId2Rank(){}

    // 可変長引数版
    template<typename... Ints>
    static int get_rank(Ints... block_ids){
        static_assert(sizeof...(Ints) == ndim,
                      "Number of block_ids must match number of Axes");

        std::array<int, ndim> b = {static_cast<int>(block_ids)...};

        int rank = 0;
        int stride = 1;

        for (int i = ndim-1; i >= 0; --i) {
            if (b[i] < 0 || b[i] >= num_blocks[i]) {
                std::cout<<"[";
                for(int j=0;j<b.size();j++){
                    std::cout<<b[i]<<" ";
                }
                std::cout<<"] "<<std::flush;
                throw std::out_of_range("block_id out of range");
            }
            rank += b[i] * stride;
            stride *= num_blocks[i];
        }
        return rank;
    }
};

#endif // BLOCK_ID2RANK_H
