#ifndef N_D_TENSOR_WITH_GHOST_CELL_H_super
#define N_D_TENSOR_WITH_GHOST_CELL_H_super

/***************************************************
 * 0. Includes / 基本型定義
 ***************************************************/
#include <cstddef>
#include <iostream>
#include <vector>
#include <array>
//#include <omp.h>
#include <mpi.h>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <fstream>
#include <cstring>
#include <utility>
#include "mpi_sendrecv_bytes.h"

using Index = int;


/***************************************************
 * 1. Slice 型定義（ユーザー API）
 ***************************************************/
struct FullSlice {};

template<Index START, Index END>
struct Slice {
    static_assert(START <= END);
    static constexpr Index START_val = START;
    static constexpr Index END_val   = END;
};


/***************************************************
 * 2. NdTensorWithGhostCell クラス宣言
 ***************************************************/
template<typename T, typename... Axes>
class NdTensorWithGhostCell {

/***************************************************
 * 2.1 Compile-time 定数・レイアウト情報
 ***************************************************/
public:
    using Value = T;
    static constexpr int N_dim = sizeof...(Axes);

    static constexpr std::array<Index, N_dim> shape = {Axes::num_grid...};
    static constexpr std::array<Index, N_dim> L_ghost_lengths = {Axes::L_ghost_length...};
    static constexpr std::array<Index, N_dim> R_ghost_lengths = {Axes::R_ghost_length...};

    static_assert([]() constexpr {
        for(int i=0;i<N_dim;++i)
            if(L_ghost_lengths[i]!=R_ghost_lengths[i]) return false;
        return true;
    }(), "ghost cell size mismatch");
    static_assert([]{
        std::array<int, sizeof...(Axes)> labels = {Axes::label...};
        for (int i = 0; i < labels.size(); ++i)
            for (int j = i+1; j < labels.size(); ++j)
                if (labels[i] == labels[j]) return false;
        return true;
    }(), "Axis labels must be unique 二つ以上の軸でlabelが被っています。");

    static constexpr int Max_label =[]()constexpr{
        std::array<Index,N_dim> labels = {Axes::label...};
        int ans=-1;
        for(int i=0;i<N_dim;i++)ans=(ans<labels[i] ?labels[i]:ans);
        return ans;
    }();

    static constexpr std::array<Index, Max_label+1> TargetAxisLabel_2_TargetDim =[]()constexpr{
        std::array<Index,Max_label+1> retval{};
        std::array<Index,N_dim> labels = {Axes::label...};
        for(int i=0;i<N_dim;i++)retval[labels[i]]=i;
        return retval;
    }();

/***************************************************
 * 2.2 メモリレイアウト（stride / offset）
 ***************************************************/
private:
    static constexpr std::array<Index, N_dim> data_shape = []{
        std::array<Index,N_dim> s{};
        for(int i=0;i<N_dim;++i)
            s[i] = shape[i] + L_ghost_lengths[i] + R_ghost_lengths[i];
        return s;
    }();

    static constexpr Index total_size = []{
        Index p=1;
        for(auto v:data_shape) p*=v;
        return p;
    }();

    static constexpr std::array<Index,N_dim> strides = []{
        std::array<Index,N_dim> st{};
        Index cur=1;
        for(int i=N_dim-1;i>=0;--i){
            st[i]=cur;
            cur*=data_shape[i];
        }
        return st;
    }();

    static constexpr Index offset = []{
        Index o=0;
        for(int i=0;i<N_dim;++i)
            o+=strides[i]*L_ghost_lengths[i];
        return o;
    }();

/***************************************************
 * 2.3 データ本体
 ***************************************************/
private:
    std::vector<T> data;

/***************************************************
 * 2.4 インデックス計算（flatten）
 ***************************************************/
private:
    template<size_t I=0, typename... Idx>
    constexpr Index flatten_index_helper(Index i, Idx... rest) const noexcept {
        if constexpr(sizeof...(rest)==0)
            return i*strides[I] + offset;
        else
            return i*strides[I] + flatten_index_helper<I+1>(rest...);
    }

    template<typename... Idx>
    constexpr Index flatten_index(Idx... idx) const noexcept {
        static_assert(sizeof...(Idx)==N_dim);
        return flatten_index_helper(idx...);
    }

/***************************************************
 * 2.5 基本アクセス API
 ***************************************************/
public:
    template<typename... Idx>
    inline T& at(Idx... idx) noexcept {
        static_assert(sizeof...(Idx) == N_dim , "引数の数が次元数と一致しません。");
        return data[flatten_index(idx...)];
    }

    template<typename... Idx>
    inline const T& at(Idx... idx) const noexcept {
        static_assert(sizeof...(Idx) == N_dim , "引数の数が次元数と一致しません。");
        return data[flatten_index(idx...)];
    }

/***************************************************
 * 2.6 set_value / slice 展開ロジック
 ***************************************************/
private:
    // set_value_helper
    // set_value_sliced_helper
    // （← 今の実装をそのまま置く）
    
    // set_value の再帰ヘルパー (物理領域のみ)
    template<typename Func, size_t Depth = 0, typename... Idx>
    void set_value_helper(const Func& func, Idx... indices) {
        
        if constexpr (Depth == sizeof...(Axes)) {
            this->at(indices...) = func(indices...);
        }
        else {
            for (Index i = 0; i < shape[Depth]; ++i) { // 物理領域 (0 .. shape-1)
                set_value_helper<Func, Depth + 1>(func, indices..., i);
            }
        }
    }


    // set_value_sliced のための再帰ヘルパー
    template<size_t Depth,typename Func, typename... Slices, typename... Idx>
    void set_value_sliced_helper(const Func& func, Idx... indices) {
        
        // 基底ケース
        if constexpr (Depth == sizeof...(Axes)) {
            this->at(indices...) = func(indices...);
        }
        // 再帰ステップ
        else {
            using CurrentSlice = std::tuple_element_t<Depth, std::tuple<Slices...>>;
            // 2. この次元の安全な境界（確保されたメモリ全体）を取得
            constexpr Index min_bound = -L_ghost_lengths[Depth];                // 例: -3
            constexpr Index max_bound = shape[Depth] + R_ghost_lengths[Depth];    // 例: 10 + 3 = 13

            if constexpr (std::is_same_v<CurrentSlice, FullSlice>) {
                // (A) FullSlice の場合: 物理領域 [0, shape[Dim]) をループ
                //for (int i = min_bound; i < max_bound; ++i) {
                for (int i = 0; i < shape[Depth]; ++i) {
                    set_value_sliced_helper<Depth+1,Func, Slices...>(func, indices..., i);
                }
            } 
            else {
                // (B) Slice<START, END> の場合:
                
                // --- (ここからがご要望の修正箇所) ---
                
                // 1. ユーザー指定の範囲を取得
                constexpr Index req_start = CurrentSlice::START_val;
                constexpr Index req_end   = CurrentSlice::END_val;
                // 3. ユーザーの要求を、安全な境界内に自動クリッピング
                
                // start_idx = max(min_bound, req_start)
                // (もし req_start が -100 なら、-3 に丸められる)
                constexpr Index start_idx = (req_start > min_bound) ? req_start : min_bound;
                
                // end_idx = min(max_bound, req_end)
                // (もし req_end が 100 なら、13 に丸められる)
                constexpr Index end_idx   = (req_end < max_bound) ? req_end : max_bound;

                // 4. クリッピングされた安全な範囲でループ
                // (もし start_idx >= end_idx ならループは実行されず、安全)
                for (Index i = start_idx; i < end_idx; ++i) {
                    set_value_sliced_helper<Depth+1,Func, Slices...>(func, indices..., i);
                }
                // --- (修正ここまで) ---
            }
        }
    }
    
public:

    template<typename Func>
    void set_value(const Func& func){
        set_value_helper(func);
    }

    /**
     * @brief 指定したスライス（部分領域）にのみ関数を適用して値を設定する
     * スライスがゴースト領域を含む場合、そこも対象となる。
     * スライスが確保されたメモリ領域を超える場合、自動的にクリッピングされる。
     */
    template<typename... Slices, typename Func>
    void set_value_sliced(const Func& func) {
        static_assert(sizeof...(Slices) == sizeof...(Axes), 
            "set_value_sliced: 次元数とスライス型の数が一致しません");
        set_value_sliced_helper<0,Func, Slices...>(func);
    }

/***************************************************
 * 2.7 Ghost cell index / buffer layout（constexpr）
 ***************************************************/
private:
    // buf_linear_index
    // ghost_size_for_axis
    // max_ghost_buffer_size
    template<int TargetDim>
    static constexpr std::array<Index, N_dim> shape_buf = [](){
        std::array<Index, N_dim> s{};
        for(int d=0; d<N_dim; ++d){
            if(d == TargetDim)
                s[d] = L_ghost_lengths[d];
            else
                s[d] = shape[d];
        }
        return s;
    }();
    
    template<int TargetDim>
    static constexpr std::array<Index, N_dim> strides_buf = [](){
        std::array<Index, N_dim> st{};
        Index cur = 1;
        for(int d=N_dim-1; d>=0; --d){
            st[d] = cur;
            cur *= shape_buf<TargetDim>[d];
        }
        return st;
    }();

    template<typename TargetAxis, bool Is_left, typename... Idx>
    static constexpr Index
    buf_linear_index(Idx... indices)
    {
        // buf_at は「受信バッファ座標系」で呼ぶ
        // idx[TargetDim] は
        //   左: 0..L-1
        //   右: shape-L..shape-1
        static_assert(sizeof...(Idx) == sizeof...(Axes));

        constexpr int TargetDim = TargetAxis::label;
        
        std::array<Index, N_dim> idx{indices...};

        if constexpr (! Is_left) {
            idx[TargetDim] -=(shape[TargetDim]-R_ghost_lengths[TargetDim]) ;
        }

        Index linear = 0;
        for(int d=0; d<N_dim; ++d)
            linear += idx[d] * strides_buf<TargetDim>[d];

        return linear;
    }

    static constexpr Index ghost_size_for_axis(int d)
    {
        Index s = L_ghost_lengths[d];
        for (int k = 0; k < sizeof...(Axes); ++k) {
            if (k != d)
                s *= shape[k];
        }
        return s;
    }

    static constexpr Index max_ghost_buffer_size = []() constexpr {
        Index m = 0;
        for (int d = 0; d < sizeof...(Axes); ++d) {
            Index s = ghost_size_for_axis(d);
            if (s > m) m = s;
        }
        return m;
    }();

/***************************************************
 * 2.8 MPI 通信ロジック
 ***************************************************/
private:
    std::vector<T> send_buf, recv_buf;

    int my_world_rank;

    void comm_init(int world_rank_){my_world_rank = world_rank_;}

    // collect_ghost_cell_helper
    // collect_ghost_cell
    // send_ghosts
    
    template<int TargetDim, bool Is_left, int Dim,typename... Indices>
    void collect_ghost_cell_helper(std::vector<T>& buf,int &buf_len,Indices... indices){
        constexpr int Ndim = sizeof...(Axes);
        if constexpr(Dim == Ndim){
            buf[buf_len++] = this->at(indices...);
        }
        else if constexpr(Dim==TargetDim){
            if constexpr(Is_left){
                for(int i=0;i<L_ghost_lengths[Dim];i++){
                    collect_ghost_cell_helper<TargetDim,Is_left,Dim+1>(buf,buf_len,indices...,i);
                }
            }
            else{
                for(int i = shape[Dim]-R_ghost_lengths[Dim];i<shape[Dim];i++){
                    collect_ghost_cell_helper<TargetDim,Is_left,Dim+1>(buf,buf_len,indices...,i);
                }
            }
        }
        else{
            for(int i=0;i<shape[Dim];i++){
                collect_ghost_cell_helper<TargetDim,Is_left,Dim+1>(buf,buf_len,indices...,i);
            }
        }
    }

    template<typename TargetAxis, bool Is_left>
    int collect_ghost_cell(std::vector<T>& buf){
        //TargetAxis のゴーストセルをsend_bufに格納する。その長さをreturn する。
        //constexpr int TargetDim = TargetAxis::label;
        constexpr int TargetDim = TargetAxisLabel_2_TargetDim[TargetAxis::label];
        int buf_len = 0;
        collect_ghost_cell_helper<TargetDim,Is_left,0>(buf,buf_len);
        return buf_len;
    }

    std::vector<int> comm_ghost_buf_sizes; 

    template<int I=0>
    void init_comm_ghost_buf_sizes(){
        if constexpr(I==N_dim)return;
        else{
            if(I==0)comm_ghost_buf_sizes.resize(N_dim);
            comm_ghost_buf_sizes[I] = collect_ghost_cell<std::tuple_element_t<I, std::tuple<Axes...>>,true>(send_buf);
            init_comm_ghost_buf_sizes<I+1>();
        }
    }

/***************************************************
 * 2.9 公開 Ghost API
 ***************************************************/
public:

    //ブロックの一方向ゴーストセルごとにデータを交換したい。
    template<typename TargetAxis,bool Is_left_send,bool Is_left_source>
    void send_and_recv_ghosts(
        int destination_world_rank,
        int source_world_rank)
    {
        //destination_world_rank==-1 のときは送信しない
        //source_world_rank==-1 のときは受信しない
        if(destination_world_rank != -1)collect_ghost_cell<TargetAxis,Is_left_send>(send_buf);
                    
        int buf_size = comm_ghost_buf_sizes[TargetAxis::label];

        int send_tag = 2 * destination_world_rank +(Is_left_send ? 1 : 0);
        int recv_tag = 2 * my_world_rank +(Is_left_source ? 1 : 0);
        mpi_sendrecv_bytes(
            send_buf,
            recv_buf,
            buf_size,
            destination_world_rank,
            source_world_rank,
            send_tag,
            recv_tag,
            MPI_COMM_WORLD);
    }

    template<typename TargetAxis,bool Is_left, typename... Idx>
    T buf_at(Idx... idx)const{
        Index id = buf_linear_index<TargetAxis,Is_left>(idx...);
        return recv_buf[id];
    }

/***************************************************
 * 2.10 I/O
 ***************************************************/
public:
    
    void save_physical(const std::string& filename) const {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) throw std::runtime_error("Failed to open file for saving");

        int64_t ndim = sizeof...(Axes);
        ofs.write(reinterpret_cast<char*>(&ndim), sizeof(int64_t));

        // write physical shape
        for (int i = 0; i < ndim; ++i) {
            int64_t s = shape[i];
            ofs.write(reinterpret_cast<char*>(&s), sizeof(int64_t));
        }
        auto writer = [&](auto&& self, auto& idxs, size_t depth) -> void {
            if (depth == ndim) {
                std::apply([&](auto... ii){
                    ofs.write(reinterpret_cast<const char*>(&this->at(ii...)), sizeof(T));
                }, idxs);
                return;
            }
            for (int i = 0; i < shape[depth]; ++i) {
                idxs[depth] = i;
                self(self, idxs, depth + 1);
            }
        };
        std::array<int, sizeof...(Axes)> idxs{};
        writer(writer, idxs, 0);
    }

    void save_physical_fast(const std::string& filename) const {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) throw std::runtime_error("Failed to open file for saving");

        int64_t ndim = N_dim;
        ofs.write(reinterpret_cast<const char*>(&ndim), sizeof(int64_t));

        for (int i = 0; i < ndim; ++i) {
            int64_t s = shape[i];
            ofs.write(reinterpret_cast<const char*>(&s), sizeof(int64_t));
        }

        // 物理領域の最初の線形位置
        constexpr int last_dim = N_dim - 1;
        const Index inner_size = shape[last_dim];

        // 外側次元の総数
        Index outer_size = 1;
        for (int d = 0; d < last_dim; ++d)
            outer_size *= shape[d];

        for (Index outer = 0; outer < outer_size; ++outer) {

            // outer を多次元 index に戻す
            Index tmp = outer;
            Index base_index = offset;

            for (int d = last_dim - 1; d >= 0; --d) {
                Index i = tmp % shape[d];
                tmp /= shape[d];
                base_index += i * strides[d];
            }

            // 物理領域の開始位置
            base_index += 0 * strides[last_dim];

            const T* ptr = &data[base_index];

            ofs.write(reinterpret_cast<const char*>(ptr),
                    sizeof(T) * inner_size);
        }
    }



    void load_physical(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) throw std::runtime_error("Failed to open file for loading");
        int64_t ndim = sizeof...(Axes);
        int64_t ndim_file;
        ifs.read(reinterpret_cast<char*>(&ndim_file), sizeof(int64_t));

        if (ndim_file != sizeof...(Axes))
            throw std::runtime_error("Dimension mismatch in load()");

        for (int i = 0; i < ndim_file; ++i) {
            int64_t s_file;
            ifs.read(reinterpret_cast<char*>(&s_file), sizeof(int64_t));
            if (s_file != shape[i])
                throw std::runtime_error("Physical shape mismatch in load()");
        }

        // clear all (ghost & physical)
        std::fill(data.begin(), data.end(),T{});

        // load only physical region
        auto reader = [&](auto&& self, auto& idxs, size_t depth) -> void {
            if (depth == ndim) {
                std::apply([&](auto... ii){
                    ifs.read(reinterpret_cast<char*>(&this->at(ii...)), sizeof(T));
                }, idxs);
                return;
            }
            for (int i = 0; i < shape[depth]; ++i) {
                idxs[depth] = i;
                self(self, idxs, depth + 1);
            }
        };

        std::array<int, sizeof...(Axes)> idxs{};
        reader(reader, idxs, 0);
    }

    void load_physical_fast(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) throw std::runtime_error("Failed to open file for loading");

        int64_t ndim_file;
        ifs.read(reinterpret_cast<char*>(&ndim_file), sizeof(int64_t));

        if (ndim_file != N_dim)
            throw std::runtime_error("Dimension mismatch in load()");

        for (int i = 0; i < ndim_file; ++i) {
            int64_t s_file;
            ifs.read(reinterpret_cast<char*>(&s_file), sizeof(int64_t));
            if (s_file != shape[i])
                throw std::runtime_error("Shape mismatch in load()");
        }

        std::fill(data.begin(), data.end(), T{});

        constexpr int last_dim = N_dim - 1;
        const Index inner_size = shape[last_dim];

        Index outer_size = 1;
        for (int d = 0; d < last_dim; ++d)
            outer_size *= shape[d];

        for (Index outer = 0; outer < outer_size; ++outer) {

            Index tmp = outer;
            Index base_index = offset;

            for (int d = last_dim - 1; d >= 0; --d) {
                Index i = tmp % shape[d];
                tmp /= shape[d];
                base_index += i * strides[d];
            }

            T* ptr = &data[base_index];

            ifs.read(reinterpret_cast<char*>(ptr),
                    sizeof(T) * inner_size);
        }
    }


/***************************************************
 * 2.11 ctor / utility
 ***************************************************/
public:
    NdTensorWithGhostCell(const int world_rank)
    {
        data.resize(total_size,T{});
        send_buf.resize(max_ghost_buffer_size,T{});
        recv_buf.resize(max_ghost_buffer_size,T{});

        init_comm_ghost_buf_sizes();
        comm_init(world_rank);
    }

    
    static constexpr int get_dimension(){return sizeof...(Axes);};

    void add(const NdTensorWithGhostCell& r){
        for(Index i=0;i<total_size;++i){
            data[i]+=r.data[i];
        }
    }
    
    
    void swap(NdTensorWithGhostCell& right) noexcept {
        std::swap(this->data, right.data);
        std::swap(send_buf, right.send_buf);
        std::swap(recv_buf, right.recv_buf);
    }

    // 非メンバ関数の swap (ADL対応)
    friend void swap(NdTensorWithGhostCell& a, NdTensorWithGhostCell& b) noexcept {
        a.swap(b);
    }
};


/**
 * @brief NdTensorWithGhostCell を構築するためのファクトリ関数
 */
template<typename T, typename... Axes>
auto make_tensor(const Axes&... axes) -> NdTensorWithGhostCell<T, Axes...> {
    return NdTensorWithGhostCell<T, Axes...>(axes...);
}

#endif
