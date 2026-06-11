#include <utility>
#include <cstddef>
#include <iostream>

// target, diff を先にする！
template <std::size_t target, int diff, typename Func, std::size_t... Is, typename... Ints>
void f_impl(Func func, std::index_sequence<Is...>, Ints... indices) {
    func((Is == target ? (indices + diff) : indices)...);
}

template <std::size_t target, int diff, typename Func, typename... Ints>
void f(Func func, Ints... indices) {
    static_assert(target < sizeof...(Ints), "target out of bounds");
    f_impl<target, diff>(func,
        std::make_index_sequence<sizeof...(Ints)>{},
        indices...);
}

void myfunc(int a, int b, int c, int d) {
    std::cout << a << " " << b << " " << c << " " << d << "\n";
}

int main() {
    for(int i=0;i<10;i++)
    f<2, 5>(myfunc, i,20,i,40);
}

