#ifndef ARG_CHANGER_H
#define ARG_CHANGER_H
#include <utility>
#include <cstddef>

namespace Utility{
namespace Arg_Changer{
template <typename Value, std::size_t target, int diff, typename Func, std::size_t... Is, typename... Ints>
Value f_impl(Func func, std::index_sequence<Is...>, Ints... indices) {
    return func((Is == target ? (indices + diff) : indices)...);
}
}
//indicesのtarget番目を+diffしてからそれをfuncに代入してくれる関数です。
template <typename Value,std::size_t target, int diff, typename Func, typename... Ints>
Value arg_changer(Func func, Ints... indices) {
    static_assert(target < sizeof...(Ints), "target out of bounds");
    return Arg_Changer::f_impl<Value,target, diff>(func,
                std::make_index_sequence<sizeof...(Ints)>{},
                indices...);
}
}
/*使用例
void myfunc(int a, int b, int c, int d) {
    std::cout << a << " " << b << " " << c << " " << d << "\n";
}

int main() {
    for(int i=0;i<10;i++)
    f<int,2, 5>(myfunc, i,20,i,40);
}
*/

#endif //ARG_CHANGER_H
