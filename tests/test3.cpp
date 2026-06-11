#include <iostream>
#include <tuple>
#include <cmath>
#include <chrono>

using Value = double;

//----------------------------------------
// 各クラス: 計算を重めに設定
//----------------------------------------
class A {
public:
    Value introduce_myself() const {
        Value s = 0;
        for (int i = 0; i < 1000; ++i)
            s += std::sin(i * 0.001) * std::cos(i * 0.002);
        return s;
    }
};

class B {
public:
    Value introduce_myself() const {
        Value s = 0;
        for (int i = 0; i < 1000; ++i)
            s += std::sin(i * 0.003) * std::cos(i * 0.004);
        return s;
    }
};

class C {
public:
    Value introduce_myself() const {
        Value s = 0;
        for (int i = 0; i < 1000; ++i)
            s += std::sin(i * 0.005) * std::cos(i * 0.006);
        return s;
    }
};

//----------------------------------------
// Pack: テンプレート展開版
//----------------------------------------
template<typename... Args>
class Pack {
private:
    std::tuple<Args...> args;

    template<int I>
    Value call_helper() const {
        auto const& obj = std::get<I>(args);
        return obj.introduce_myself();
    }

public:
    Pack(Args... a) : args(std::move(a)...) {}

    template<int I>
    Value introduce_someone() const {
        return call_helper<I>();
    }
};

//----------------------------------------
// メイン
//----------------------------------------
int main() {
    A a; B b; C c;
    Pack<A,B,C> p(a,b,c);
    constexpr int N = 100000;  // 繰り返し回数
    volatile Value sum1 = 0.0;
    volatile Value sum2 = 0.0;

    //----------------------------------------
    // case 1: テンプレート展開（静的呼び出し）
    //----------------------------------------
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        sum1 += p.introduce_someone<0>();
        sum1 += p.introduce_someone<1>();
        sum1 += p.introduce_someone<2>();
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    //----------------------------------------
    // case 2: 愚直な for + if 文（ランタイム分岐）
    //----------------------------------------
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (j == 0) sum2 += a.introduce_myself();
            else if (j == 1) sum2 += b.introduce_myself();
            else sum2 += c.introduce_myself();
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    //----------------------------------------
    // 結果表示
    //----------------------------------------
    double elapsed1 = std::chrono::duration<double>(t2 - t1).count();
    double elapsed2 = std::chrono::duration<double>(t4 - t3).count();

    std::cout << "=== Result ===\n";
    std::cout << "Template version : " << elapsed1 << " s\n";
    std::cout << "Naive for-if ver : " << elapsed2 << " s\n";
    std::cout << "Speedup ratio    : " << (elapsed2 / elapsed1) << "x\n";
    std::cout << "sum1=" << sum1 << " sum2=" << sum2 << std::endl;

    return 0;
}

