#include <iostream>
#include <mpi.h>

#include "../current.h"
#include "../../pack.h"
#include "../axis.h"

using Axis_x_  = Axis<0,4,1,3>;
using Axis_y_  = Axis<1,5,1,3>;
using Axis_vx  = Axis<2,2,1,3>;
using Axis_vy  = Axis<3,2,4,3>;

using VeloPack = Pack<Axis_vx,Axis_vy>;
using CurrentType = Current<double, VeloPack, Axis_x_, Axis_y_>;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    CurrentType current;

    constexpr int velo_blocks =
        Axis_vx::num_blocks * Axis_vy::num_blocks;

    if(world_size % velo_blocks != 0){
        if(world_rank == 0){
            std::cerr << "world_size must be multiple of velo_blocks\n";
        }
        MPI_Finalize();
        return 0;
    }

    int my_velo_rank = world_rank % velo_blocks;

    // 各ランクでローカル電流をセット
    current.set_value([&](int ix, int iy){
        return (double)my_velo_rank;
    });

    // グローバル電流計算
    current.compute_global_current();

    // rank0だけ表示
    if(world_rank == 0){
        std::cout << "global current:\n";
        for(int i=0;i<Axis_x_::num_grid;i++){
            for(int j=0;j<Axis_y_::num_grid;j++){
                std::cout << current.global_at(i,j) << " ";
            }
            std::cout << "\n";
        }

        double expect = velo_blocks * (velo_blocks - 1) / 2.0;
        std::cout << "expected value = " << expect << std::endl;
    }

    MPI_Finalize();
}
