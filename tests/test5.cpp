#include <utility>
#include <iostream>

// g<I>(x) を呼び出し、0..N まで展開して f(...) に渡す
template <typename Func1, typename Func2, std::size_t... Is>
void func_impl(Func1 f1, Func2 f2, double x, std::index_sequence<Is...>) {
    f1(f2.template operator()<Is>(x)...);
}

template<typename Func1, typename Func2, std::size_t N>
void func(Func1 f1, Func2 f2, double x) {
    func_impl(f1, f2, x, std::make_index_sequence<N+1>{});
}
struct F {
    template<typename... Args>
    void operator()(Args... args) const {
        ((std::cout << args << " "), ...);
        std::cout << "\n";
    }
};

// g<i>(x) を提供する functor
struct G {
    template<std::size_t I>
    double operator()(double x) const {
        return x + I;
    }
};

int main() {
    func<F,G,4>(F{}, G{}, 10.0);  
    // g<0>(10)=10, g<1>(10)=11, ..., g<4>(10)=14
    // 出力: 10 11 12 13 14
}