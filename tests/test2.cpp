#include <iostream>
#include <tuple>

class A {
public:
    void introduce_myself() { std::cout << "a"; }
};

class B {
public:
    void introduce_myself() { std::cout << "b"; }
};

class C {
public:
    void introduce_myself() { std::cout << "c"; }
};

template<typename... Args>
class Pack {
private:
    std::tuple<Args...> args;  // 引数を保持

    // I番目の要素を取得して呼び出すヘルパ
    template<int I>
    void call_helper() {
        auto& obj = std::get<I>(args);
        obj.introduce_myself();
    }

public:
    Pack(Args... a) : args(std::move(a)...) {}

    template<int I>
    void introduce_someone() {
        call_helper<I>();
    }
};

int main() {
    A a; B b; C c;
    Pack<A,B,C> p(a,b,c);

    p.introduce_someone<0>();  // → a
    p.introduce_someone<1>();  // → b
    p.introduce_someone<3>();  // → c
    std::cout << std::endl;
}