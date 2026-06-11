//
// n_d_tensor_with_ghost_cell.h
//
#ifndef N_D_TENSOR_WITH_GHOST_CELL_H
#define N_D_TENSOR_WITH_GHOST_CELL_H

#include <iostream>
#include <vector>
#include <array>
#include <functional>
#include <omp.h>
#include <cmath>
#include <tuple>      // スライス機能のために追加
#include <type_traits> // スライス機能のために追加
#include <fstream>
#include <cstring>
#include <utility>

using Index = int;

// --- スライス用ヘルパー型 ---

/**
 * @brief スライス指定: この次元の全物理領域 (:) を対象とすることを示すタグ
 */
struct FullSlice {};

/**
 * @brief スライス指定: [START, END) の半開区間を対象とすることを示すタグ
 * (Pythonの [START:END] と同じ)
 */
template<Index START, Index END>
struct Slice {
    static_assert(START <= END, "Slice: START must be less than or equal to END");
    static constexpr Index START_val = START;
    static constexpr Index END_val = END;
};
// --- (ここまで) ---


template<typename T,typename... Axes>
class NdTensorWithGhostCell {
public:
    // 軸ごとの長さをコンパイル時に収集
    static constexpr std::array<Index, sizeof...(Axes)> shape = {Axes::num_grid...};
    // 軸ごとのマイナス側ゴーストセル数をコンパイル時に収集
    static constexpr std::array<Index, sizeof...(Axes)> L_ghost_lengths = {Axes::L_ghost_length...};
    // 軸ごとのプラス側ゴーストセル数をコンパイル時に収集
    static constexpr std::array<Index, sizeof...(Axes)> R_ghost_lengths = {Axes::R_ghost_length...};

    static constexpr std::array<Index,sizeof...(Axes)> data_shape 
        = [](){
            std::array<Index, sizeof...(Axes)> s = {};
            for(int i=0;i<(int)sizeof...(Axes);++i){
                s[i]=shape[i]+L_ghost_lengths[i]+R_ghost_lengths[i];
            }
            return s;
        } ();
private:
    // 総要素数をコンパイル時計算
    static constexpr Index total_size = []() constexpr {
        Index prod = 1;
        for (auto s : data_shape) prod *= s;
        return prod;
    }();

    std::vector<T> data;

    // ストライドをコンパイル時に計算して配列に格納
    static constexpr std::array<Index, sizeof...(Axes)> strides = []() constexpr {
        std::array<Index, sizeof...(Axes)> s = {};
        Index current_stride = 1;
        
        for (int i = (int)sizeof...(Axes) - 1; i >= 0; --i) {
            s[i] = current_stride;
            current_stride *= data_shape[i];
        }
        return s;
    }();

    // オフセットをコンパイル時に計算 (物理インデックス(0,0..)が data[] のどこか)
    static constexpr Index offset = []() constexpr {
        Index ret_val = 0;
        for(int i=0;i<sizeof...(Axes);++i){
            ret_val+= strides[i]*L_ghost_lengths[i];
        }
        return ret_val;
    }();

    // flatten_index_helper
    template<size_t I = 0, typename... IdT>
    constexpr int flatten_index_helper(Index i, IdT... rest) const noexcept {
        if constexpr (sizeof...(rest) == 0) {
            return i * strides[I]+offset; 
        } else {
            return i * strides[I] + flatten_index_helper<I + 1>(rest...);
        }
    }

    // 外部から呼び出す flatten_index
    template<typename... Idx>
    constexpr int flatten_index(Idx... indices) const noexcept {
        static_assert(sizeof...(Idx) == sizeof...(Axes));
        return flatten_index_helper(indices...);
    }


    // set_value の再帰ヘルパー (物理領域のみ)
    template<typename Func, size_t Depth = 0, typename... Idx>
    void set_value_helper(Func func, Idx... indices) {
        
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
    void set_value_sliced_helper(Func func, Idx... indices) {
        
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
                for (int i = min_bound; i < max_bound; ++i) {
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
    NdTensorWithGhostCell(const Axes& ...args){
        data.resize(total_size);
    }
    NdTensorWithGhostCell(){
        data.resize(total_size);
    }

    void add(const NdTensorWithGhostCell& r){
        for(Index i=0;i<total_size;++i){
            data[i]+=r.data[i];
        }
    }
    
    // at() アクセサ
    template<typename... Idx>
    inline T& at(Idx... indices) noexcept {
        static_assert(sizeof...(Idx) == sizeof...(Axes));
        return data[flatten_index(indices...)];
    }

    template<typename... Idx>
    inline const T& at(Idx... indices) const noexcept {
        static_assert(sizeof...(Idx) == sizeof...(Axes));
        return data[flatten_index(indices...)];
    }

    /**
     * @brief 物理領域全体に関数を適用して値を設定する
     */
    template<typename Func>
    void set_value(Func func){
        set_value_helper(func);
    }

    /**
     * @brief 指定したスライス（部分領域）にのみ関数を適用して値を設定する
     * スライスがゴースト領域を含む場合、そこも対象となる。
     * スライスが確保されたメモリ領域を超える場合、自動的にクリッピングされる。
     */
    template<typename... Slices, typename Func>
    void set_value_sliced(Func func) {
        static_assert(sizeof...(Slices) == sizeof...(Axes), 
            "set_value_sliced: 次元数とスライス型の数が一致しません");
        
        set_value_sliced_helper<0,Func, Slices...>(func);
    }

    static constexpr int get_dimension(){return sizeof...(Axes);};

    

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
        std::fill(data.begin(), data.end(), T{0});

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

    void swap(NdTensorWithGhostCell& right) noexcept {
        std::swap(this->data, right.data);
    }

    // 非メンバ関数の swap (ADL対応)
    friend void swap(NdTensorWithGhostCell& a, NdTensorWithGhostCell& b) noexcept {
        a.swap(b);
    }

};

// クラステンプレート引数の推論補助 (CTAD)
template <typename T, typename... Axes>
NdTensorWithGhostCell(const Axes&...) -> NdTensorWithGhostCell<T, Axes...>;

/**
 * @brief NdTensorWithGhostCell を構築するためのファクトリ関数
 */
template<typename T, typename... Axes>
auto make_tensor(const Axes&... axes) -> NdTensorWithGhostCell<T, Axes...> {
    return NdTensorWithGhostCell<T, Axes...>(axes...);
}

#endif // N_D_TENSOR_WITH_GHOST_CELL_H
/*

使用例
#include "axis.h"
#include "n_d_tensor_with_ghost_cell.h"
int main(){
    Axis<0,100> vx,vy,vz;
    Axis<1,1000> x;

    //Axisクラスをn個入力するとgはn次元テンソルになります。
    //↓の場合は1000*100*100*100の４次元テンソル
    auto g = make_tensor_with_ghost_cell<double>(x, vx, vy, vz);

    for(int i=0;i<1000;++i){
        for(int j=0;j<100;++j){
            for(int k=0;k<100;++k){
                for(int l=0;l<100;++l){
                    g(i,j,k,l) = (double)i*(double)j*(double)k*(double)l;
                }
            }
        }
    }

    //また、Axis class の L_ghost_lengthが５に設定されている場合は、
    g(-3,12,-1,300) = 3.;
    //などのアクセスが許可されます。


    // 3次元テンソル (double型) を作成
    using AxisX = Axis<0, 10, 2, 2>; // 物理: 10 (0..9),  ゴースト: 2 (計 14)
    using AxisY = Axis<1, 8,  2, 2>; // 物理: 8  (0..7),  ゴースト: 2 (計 12)
    using AxisZ = Axis<2, 6,  2, 2>; // 物理: 6  (0..5),  ゴースト: 2 (計 10)
    auto f = make_tensor<double>(AxisX{}, AxisY{}, AxisZ{});

    // 1. set_value (全範囲) を使って、全物理領域を 1.0 で初期化
    std::cout << "1. 全物理領域 ([0..9], [0..7], [0..5]) を 1.0 で初期化中..." << std::endl;
    f.set_value([](int i, int j, int k){
        return 1.0;
    });

    // 2. set_value_sliced を使って、指定した部分領域の値を 99.0 に上書き
    // X軸: [3, 6) (つまり 3, 4, 5)
    // Y軸: 全範囲 (:)
    // Z軸: [1, 4) (つまり 1, 2, 3)
    std::cout << "2. スライス ([3..5], :, [1..3]) の値を 99.0 に上書き中..." << std::endl;
    f.set_value_sliced<
        Slice<3, 6>,  // X軸スライス
        FullSlice,    // Y軸スライス
        Slice<1, 4>   // Z軸スライス
    >([](int i, int j, int k){
        // ラムダにはスライスされた物理インデックス (i, j, k) が渡される
        return 99.0;
    });

    // 3. ゴースト領域に手動で値を書き込む (境界条件のシミュレート)
    std::cout << "3. ゴースト領域に手動で値を書き込み中..." << std::endl;
    f.at(-1, 0, 0) = -1.0; // 左側ゴースト
    f.at(10, 0, 0) = -1.0; // 右側ゴースト (物理サイズが10なのでインデックス10はゴースト)


    // 4. 結果の確認
    std::cout << "\n--- 結果の確認 ---" << std::endl;
    std::cout << std::fixed << std::setprecision(1);

    // (A) ゴースト領域の値
    std::cout << "(A) ゴースト領域の値: f.at(-1, 0, 0) = " << f.at(-1, 0, 0) << " (期待値: -1.0)" << std::endl;

    // (B) set_value で初期化された領域 (スライスの外側)
    std::cout << "(B) スライスの外側:  f.at(1, 1, 1) = " << f.at(1, 1, 1) << " (期待値: 1.0)" << std::endl;
    std::cout << "(C) スライスの外側:  f.at(4, 4, 4) = " << f.at(4, 4, 4) << " (期待値: 1.0)" << std::endl;

    // (D) set_value_sliced で上書きされた領域 (スライスの内側)
    std::cout << "(D) スライスの内側:  f.at(3, 1, 1) = " << f.at(3, 1, 1) << " (期待値: 99.0)" << std::endl;
    std::cout << "(E) スライスの内側:  f.at(4, 4, 2) = " << f.at(4, 4, 2) << " (期待値: 99.0)" << std::endl;
    std::cout << "(F) スライスの内側:  f.at(5, 7, 3) = " << f.at(5, 7, 3) << " (期待値: 99.0)" << std::endl;
    
    // (G) スライスの境界チェック (X=6 は範囲外 [3, 6) なので 1.0 のまま)
    std::cout << "(G) スライスの境界外: f.at(6, 1, 1) = " << f.at(6, 1, 1) << " (期待値: 1.0)" << std::endl;
    
    return 0;
}
*/
