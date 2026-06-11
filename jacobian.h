#ifndef JACOBIAN_H
#define JACOBIAN_H
#include <tuple>
#include <string>
#include <type_traits> // 型の確認用
template<typename... Xi_diff_x>
class Jacobian
//Jacobian class 廃止予定　class Packの入れ子で実装した方がまとまる。
{
private:
    using Table = std::tuple<const Xi_diff_x&...>;;
    const Table elements;
    static constexpr int num_args = sizeof...(Xi_diff_x);
    // 依存コンテキストで平方数チェック ＆ sqrt(n) を決める
    static constexpr int num_label = []{
        for(int i = 0; i <= 20; ++i) {
            if(i * i == num_args) return i;
        }
        return 0;
    }();
    static_assert(num_label*num_label==num_args,"template引数の数は平方数である必要があります");
public:
    Jacobian(const Xi_diff_x&... args):
        elements(args...)
    {}
    template<int I,int J>
    const auto& get_element()const{
        return std::get<num_label*I+J>(elements);
    }

    template<int I,int J>
    using element_t = typename std::tuple_element_t<num_label*I+J, Table>;
};
#endif //JACOBIAN_H