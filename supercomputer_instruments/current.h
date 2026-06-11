#ifndef CURRENT_H
#define CURRENT_H


#include "n_d_tensor.h"
#include "mpi_sendrecv_bytes.h"
#include <mpi.h>
#include <vector>

template<typename T,typename VeloPack,typename... Axes>
class Current{
private:
    int my_world_rank;
    NdTensor<T, Axes...> tensor;
    NdTensor<T, Axes...> global;
    NdTensor<T, Axes...> recv_buf;

    std::vector<int> ranks_to_com;

    static constexpr int VeloDim = VeloPack::get_num_objects();

    template<int I>
    using VeloAxes = typename VeloPack::template element<I>;

    template<int Depth>
    static constexpr int calc_velo_block_num_helper(){
        if constexpr(Depth==VeloDim){
            return 1;
        }
        else{
            return VeloAxes<Depth>::num_blocks*calc_velo_block_num_helper<Depth+1>();
        }
    }

    static constexpr int velo_block_num = calc_velo_block_num_helper<0>();

    static_assert(
        []()constexpr{
            int x = 1;
            for(int i=0;i<30;i++){
                if(velo_block_num==x)return true;
                x*=2;
            }
            return false;
        }(),
        "速度空間のブロック数は２の階乗である必要があります。"
        "Number of blocks in velocity space must be power of 2."
    );

    //メモリ分散型の並列プログラミングでは、
    //ローカル電流を足し合わせる必要がある。
    //その足し合わせる作業では、
    //log_velo_block_num回の通信が必要である。
    static constexpr int log_velo_block_num = []()constexpr{
        int x = 1;
        for(int i=0;i<30;i++){
            if(velo_block_num==x)return i;
            x*=2;
        }
        return -1;
    }();

public:
    static constexpr int get_log_velo_block_num(){
        return log_velo_block_num;
    }
    Current(int my_world_rank):
        my_world_rank(my_world_rank)
    {
        ranks_to_com.resize(log_velo_block_num);

        std::vector<int> velo_ranks_to_comm(log_velo_block_num);
        int my_velo_rank = my_world_rank % velo_block_num;
        const int my_real_rank = my_world_rank / velo_block_num;
        int now = my_velo_rank;
        /*int pow = 1;
        for(int i=0;i<log_velo_block_num;i++){
            if(now%(pow*2)<pow){
                now+=pow;
            }
            else{
                now-=pow;
            }
            velo_ranks_to_comm[i]=now;
            pow*=2;
        }*/
        int pow = 1;
        for(int i=0;i<log_velo_block_num;i++){
            int target = my_velo_rank; // 毎回自分のランクをベースにする
            if(target % (pow*2) < pow){
                target += pow;
            }
            else{
                target -= pow;
            }
            velo_ranks_to_comm[i] = target;
            pow *= 2;
        }
        
        for(int i=0;i<log_velo_block_num;i++){
            // 1 << i は 2のi乗 (pow) を意味します。
            // ^ はビットごとの排他的論理和(XOR)で、指定したビットのみを反転させます。
            velo_ranks_to_comm[i] = my_velo_rank ^ (1 << i);
        }

        for(int i=0;i<log_velo_block_num;i++){
            ranks_to_com[i]
                = my_real_rank * velo_block_num
                    + velo_ranks_to_comm[i];
        }
    }    
    

    void compute_global_current(MPI_Comm comm = MPI_COMM_WORLD)
    {
        // global = local で初期化
        global.data = tensor.data;

        const int count = global.data.size();


        for(int step = 0; step < log_velo_block_num; ++step){
            int partner = ranks_to_com[step];

            mpi_sendrecv_bytes(
                global.data,        // send
                recv_buf.data,      // recv
                count,
                partner, partner,
                0, 0,
                comm
            );

            //#pragma omp parallel for
            for(int i = 0; i < count; ++i){
                global.data[i] += recv_buf.data[i];
            }
        }
    }

    void clear(){
        const int data_size = tensor.data.size();
        for(int i=0;i<data_size;i++){
            tensor.data[i]= T{};
        }
    }

    //グローバル電流 
    //グローバル電流は外から書き換える必要ないのでconst ref のみ
    template<typename... Idx>
    inline const T& global_at(Idx... indices) noexcept {
        static_assert(sizeof...(Idx) == sizeof...(Axes));
        return global.at(indices...);
    }


    //ローカル電流
    template<typename... Idx>
    inline T& at(Idx... indices) noexcept {
        static_assert(sizeof...(Idx) == sizeof...(Axes));
        return tensor.at(indices...);
    }

    //ローカル電流
    template<typename... Idx>
    inline const T& at(Idx... indices)const noexcept {
        static_assert(sizeof...(Idx) == sizeof...(Axes));
        return tensor.at(indices...);
    }

    template<typename Func>
    void set_value(const Func& func){
        // ヘルパーを初期呼び出し (インデックスは空)
        tensor.set_value(func);
    }
    static constexpr int get_dimension(){return sizeof...(Axes);};

    static constexpr std::array<int, sizeof...(Axes)> shape = {Axes::num_grid...};

};

#endif //CURRENT_H