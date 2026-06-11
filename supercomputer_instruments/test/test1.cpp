#include "../axis.h"
#include "../axis_instantiator.h"
#include "../block_id2rank.h"
#include <iostream>
using Axis_x_ = Axis<0,10,3,3>;
using Axis_y_ = Axis<0,10,4,3>;
using Axis_z_ = Axis<0,10,5,3>;
int main(){
    int rank =55;
    const auto [axis_x,axis_y,axis_z] = axis_instantiator<Axis_x_,Axis_y_,Axis_z_>(rank);
    BlockId2Rank blockid_2_rank(axis_x,axis_y,axis_z);

    std::cout<<axis_x.block_id<<"\n";
    std::cout<<axis_y.block_id<<"\n";
    std::cout<<axis_z.block_id<<"\n";

    std::cout<<"\n";
    std::cout<<blockid_2_rank.get_rank(axis_x.block_id,axis_y.block_id,axis_z.block_id)<<"\n";
    std::cout<<"\n";
}