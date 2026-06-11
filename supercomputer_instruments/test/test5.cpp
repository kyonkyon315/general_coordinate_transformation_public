#include "../axis.h"
#include "../n_d_tensor_with_ghost_cell.h"
#include "../../utils/Timer.h"
#include <iostream>

const int NX = 16;
const int NY = 16;
const int NZ = 16;

// LABEL, LENGTH, NUM_BLOCKS, LEN_GHOST
using X = Axis<0, NX, 1, 3>;
using Y = Axis<1, NY, 1, 3>;
using Z = Axis<2, NZ, 1, 3>;

using Tensor = NdTensorWithGhostCell<float, X, Y, Z>;

int main(){
    Tensor tensor;

    volatile float sink = 0.0; // 最適化防止

    Timer timer;

    // ========================
    // テンプレ再帰版
    // ========================
    for(int it=0; it<5; ++it){
        tensor.set_value([](int i,int j,int k){
            return float(i + j + k);
        });
    }

    timer.start();
    for(int it=0; it<50000; ++it){
        tensor.set_value([](int i,int j,int k){
            return float(i + j + k);
        });
    }
    timer.stop();
    std::cout << "set_value : " << timer << "\n";

    sink += tensor.at(0,0,0);

    // ========================
    // 生 for ループ版
    // ========================
    timer.start();
    for(int it=0; it<50000; ++it){
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NY; ++j) {
                for (int k = 0; k < NZ; ++k) {
                    tensor.at(i,j,k) = float(i + j + k);
                }
            }
        }
    }
    timer.stop();
    std::cout << "baseline  : " << timer << "\n";

    sink += tensor.at(1,1,1);
    std::cerr << "sink=" << sink << "\n";

    return 0;
}
