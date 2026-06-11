#include <vector>
#include <array>
//#include <omp.h>
#include <cmath>

template<typename T,typename... Axes>
class NdTensor {
public:
    // 軸ごとの長さをコンパイル時に収集
    static constexpr std::array<int, sizeof...(Axes)> shape = {Axes::num_grid...};
    std::vector<T> data;

private:
    // 総要素数をコンパイル時計算
    static constexpr int total_size = []() constexpr {
        int prod = 1;
        for (auto s : shape) prod *= s;
        return prod;
    }();


    // (NEW) ストライドをコンパイル時に計算して配列に格納
    // C-style (row-major) のストライド計算
    // (i,j,k) (shape={S0,S1,S2}) の場合、strides は {S1*S2, S2, 1} となる
    static constexpr std::array<int, sizeof...(Axes)> strides = []() constexpr {
        std::array<int, sizeof...(Axes)> s = {};
        int current_stride = 1;
        
        // 末尾の次元から計算 (k -> j -> i)
        for (int i = (int)sizeof...(Axes) - 1; i >= 0; --i) {
            s[i] = current_stride;
            current_stride *= shape[i];
        }
        return s;
    }(); // IIFE (即時実行関数) によりコンパイル時に計算が完了

    // (REVISED) 新しい flatten_index_helper (ドット積)
    // (i, j, k) と (s[0], s[1], s[2]) のドット積を計算する
    template<size_t I = 0, typename... IdT>
    constexpr int flatten_index_helper(int i, IdT... rest) const noexcept {
        if constexpr (sizeof...(rest) == 0) {
            return i * strides[I]; // 最後のインデックス (例: k * s[2])
        } else {
            // 現在のインデックス * ストライド + 残りの計算
            // 例: i * s[0] + flatten_index_helper<1>(j, k)
            return i * strides[I] + flatten_index_helper<I + 1>(rest...);
        }
    }

    // (REVISED) 外部から呼び出す flatten_index
    template<typename... Idx>
    constexpr int flatten_index(Idx... indices) const noexcept {
        static_assert(sizeof...(Idx) == sizeof...(Axes));
        return flatten_index_helper(indices...);
    }


    // set_value の再帰ヘルパー 
    template<typename Func, size_t Dim = 0, typename... Idx>
    void set_value_helper(const Func& func, Idx... indices) {
        
        // 基底ケース: 全ての次元のインデックスが揃った
        if constexpr (Dim == sizeof...(Axes)) {
            // operator() を呼び出して値を設定
            this->at(indices...) = func(indices...);
        }
        // 再帰ステップ:
        else {
            for (int i = 0; i < shape[Dim]; ++i) {
                set_value_helper<Func, Dim + 1>(func, indices..., i);
            }
        }
    }


public:
    NdTensor(const Axes& ...args){
        data.resize(total_size);
    }
    NdTensor(){
        data.resize(total_size);
    }
    
    // operator() は (シグネチャは) 変更なし
    // 内部で呼び出す flatten_index が最適化されている
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

    template<typename Func>
    void set_value(const Func& func){
        // ヘルパーを初期呼び出し (インデックスは空)
        set_value_helper(func);
    }
    static constexpr int get_dimension(){return sizeof...(Axes);};
};

template<typename T, typename... Axes>
auto make_tensor(const Axes&... axes) -> NdTensor<T, Axes...> {
    return NdTensor<T, Axes...>(axes...);
}

/*使用例

template<int LENGTH>
class Axis {
    private:
    int val;
    public:
    static constexpr int length = LENGTH;
    int operator=(int r){
        return val=r;
    }
    int operator()(){
        return val;
    }
    int get_length(){
        return length;
    }
};

int main(){
    Axis<100> vx,vy,vz;
    Axis<1000> x;

    //Axisクラスをn個入力するとfはn次元テンソルになります。
    //↓の場合は1000*100*100*100の４次元テンソル
    auto f = make_tensor<double>(xi, vx, vy, vz);

    for(int i=0;i<1000;++i){
        for(int j=0;j<100;++j){
            for(int k=0;k<100;++k){
                for(int l=0;l<100;++l){
                    f(i,j,k,l) = (double)i*(double)j*(double)k*(double)l;
                }
            }
        }
    }
    return 0;
}
*/



