#include <mpi.h>
#include <iostream>
#include <string>
#include "../n_d_tensor_with_ghost_cell.h"
#include "../axis.h"

#include "../axis_instantiator.h"
#include "../block_id2rank.h"
// 1D
using X = Axis<0,10,3,2>;
using Y = Axis<1,10,2,2>;
#include <sstream>
#include <iomanip>

std::string int_to_string_5digit(int x)
{
    std::ostringstream oss;
    oss << std::setw(5) << std::setfill(' ') << x;
    return oss.str();
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 6) {
        if (rank == 0)
            std::cerr << "This test requires exactly 6 MPI ranks\n";
        MPI_Finalize();
        return 0;
    }

    const auto [axis_x, axis_y] = axis_instantiator<X,Y>(rank);
    BlockId2Rank blockid_2_rank(axis_x,axis_y);

    NdTensorWithGhostCell<int,X,Y> tensor;

    tensor.set_value(
        [&](int i,int j){
            return (axis_x.block_id*axis_x.num_grid+i)*axis_y.num_blocks*axis_y.num_grid
            +(axis_y.block_id*axis_y.num_grid+j);
        }
    );

    MPI_Barrier(MPI_COMM_WORLD);

    std::string shape_before = "[Rank "+std::to_string(rank)+"] physical before exchange:\n";

    for(int i=-X::L_ghost_length;i<X::num_grid+X::R_ghost_length;++i){
        for(int j=-Y::L_ghost_length;j<Y::num_grid+Y::R_ghost_length;++j){
            shape_before+=(int_to_string_5digit(tensor.at(i,j))+" ");
        }
        shape_before+="\n";
    }
    shape_before+="\n";

    std::cout<<shape_before;



    int dst_rank,src_rank;
    int dst_block_id_x, dst_block_id_y;
    int src_block_id_x, src_block_id_y;
    dst_block_id_x = (axis_x.num_blocks+axis_x.block_id+1)%axis_x.num_blocks;
    src_block_id_x = (axis_x.num_blocks+axis_x.block_id-1)%axis_x.num_blocks;
    dst_block_id_y = (axis_y.num_blocks+axis_y.block_id+0)%axis_y.num_blocks;
    src_block_id_y = (axis_y.num_blocks+axis_y.block_id-0)%axis_y.num_blocks;
    dst_rank = blockid_2_rank.get_rank(dst_block_id_x,dst_block_id_y);
    src_rank = blockid_2_rank.get_rank(src_block_id_x,src_block_id_y);
    
    tensor.send_ghosts<X,false,false>(dst_rank,src_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    for(int i=X::num_grid-X::R_ghost_length;i<X::num_grid;i++){
        for(int j=0;j<Y::num_grid;j++){
            tensor.at(i-X::num_grid,j)=tensor.buf_at<X,false>(i,j);
        }
    }

    dst_block_id_x = (axis_x.num_blocks+axis_x.block_id+0)%axis_x.num_blocks;
    src_block_id_x = (axis_x.num_blocks+axis_x.block_id-0)%axis_x.num_blocks;
    dst_block_id_y = (axis_y.num_blocks+axis_y.block_id+1)%axis_y.num_blocks;
    src_block_id_y = (axis_y.num_blocks+axis_y.block_id-1)%axis_y.num_blocks;
    dst_rank = blockid_2_rank.get_rank(dst_block_id_x,dst_block_id_y);
    src_rank = blockid_2_rank.get_rank(src_block_id_x,src_block_id_y);

    tensor.send_ghosts<Y,false,false>(dst_rank,src_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int i=0;i<X::num_grid;i++){
        for(int j=Y::num_grid-Y::R_ghost_length;j<Y::num_grid;j++){
            tensor.at(i,j-Y::num_grid)=tensor.buf_at<Y,false>(i,j);
        }
    }

    dst_block_id_x = (axis_x.num_blocks+axis_x.block_id-1)%axis_x.num_blocks;
    src_block_id_x = (axis_x.num_blocks+axis_x.block_id+1)%axis_x.num_blocks;
    dst_block_id_y = (axis_y.num_blocks+axis_y.block_id-0)%axis_y.num_blocks;
    src_block_id_y = (axis_y.num_blocks+axis_y.block_id+0)%axis_y.num_blocks;
    dst_rank = blockid_2_rank.get_rank(dst_block_id_x,dst_block_id_y);
    src_rank = blockid_2_rank.get_rank(src_block_id_x,src_block_id_y);

    tensor.send_ghosts<X,true,true>(dst_rank,src_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    for(int i=0; i < X::L_ghost_length; i++){
        for(int j=0;j<Y::num_grid;j++){
            tensor.at(X::num_grid+i,j)=tensor.buf_at<X,true>(i,j);
        }
    }

    
    dst_block_id_x = (axis_x.num_blocks+axis_x.block_id-0)%axis_x.num_blocks;
    src_block_id_x = (axis_x.num_blocks+axis_x.block_id+0)%axis_x.num_blocks;
    dst_block_id_y = (axis_y.num_blocks+axis_y.block_id-1)%axis_y.num_blocks;
    src_block_id_y = (axis_y.num_blocks+axis_y.block_id+1)%axis_y.num_blocks;
    dst_rank = blockid_2_rank.get_rank(dst_block_id_x,dst_block_id_y);
    src_rank = blockid_2_rank.get_rank(src_block_id_x,src_block_id_y);


    tensor.send_ghosts<Y,true,true>(dst_rank,src_rank);
    
    MPI_Barrier(MPI_COMM_WORLD);

    for(int i=0; i < X::num_grid; i++){
        for(int j=0 ;j<Y::L_ghost_length;j++){
            tensor.at(i,Y::num_grid+j)=tensor.buf_at<Y,true>(i,j);
        }
    }

    // ----------------------------------
    // 4. 結果表示
    // ----------------------------------
    std::string shape_after = "[Rank "+std::to_string(rank)+"] physical after exchange:\n";

    for(int i=-X::L_ghost_length;i<X::num_grid+X::R_ghost_length;++i){
        for(int j=-Y::L_ghost_length;j<Y::num_grid+Y::R_ghost_length;++j){
            shape_after+=(int_to_string_5digit(tensor.at(i,j))+" ");
        }
        shape_after+="\n";
    }
    shape_after+="\n";

    std::cout<<shape_after;
    MPI_Finalize();

    return 0;
}


