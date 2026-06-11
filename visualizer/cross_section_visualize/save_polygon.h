#ifndef SAVE_POLYGON_H
#define SAVE_POLYGON_H

#include <array>

#include "../../utils/bin_saver.h"
template<typename Integers,typename Operators,typename... Axes>
inline void save_polygon(const Operators& operators,BinSaver& bin_saver,const std::string& dist_filename){
    static constexpr std::array num_blocks = {(Axes::num_block)...};
    static constexpr int num_block = []()constexpr{
        int retv=1;
        for(int i=0;i<num_blocks.size();i++)retv*=num_blocks[i];
        return retv;
    };

    BinSaver polygon_file;
    for(int world_rank = 0;world_rank<num_block;++world_rank){

        std::tuple<Axes...> axes = axis_instantiator<Axes...>(world_rank);
    
        const std::string filename = "/LARGE1/gr20001/b39211/Documents/general_coodinate_transformation/whistler/dist_func/"
                 "step" + std::to_string(step)+"_"
                 + std::to_string(axis_z_.block_id) + "_"
                 + std::to_string(axis_vr.block_id) + "_"
                 + std::to_string(axis_vt.block_id) + "_"
                 + std::to_string(axis_vp.block_id)
                 + ".bin";
        //Integers<0,0,-1,-1>
        save_polygon<Integers<0,0,-1,-1>>(operators,polygon_file);
    }
}

#endif //SAVE_POLYGON_H