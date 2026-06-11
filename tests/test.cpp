#include <iostream>
#include <type_traits> // std::is_same_v, std::true_type, std::false_type
#include <cmath>
// --------------------------------------------------
// 1. ユーザー定義の A
// --------------------------------------------------
template<typename... Args>
class A {};

// --------------------------------------------------
// 2. ヘルパー1: 型がパック内に存在するかを判定 (is_in_pack)
// --------------------------------------------------
// C++17 の畳み込み式 (fold expression) を使った最も簡単な方法
template<typename T, typename... Pack>
constexpr bool is_in_pack = (std::is_same_v<T, Pack> || ...);
// (std::is_same_v<T, P1> || std::is_same_v<T, P2> || ...) と同じ意味

// --------------------------------------------------
// 3. ヘルパー2: T の形を判定し、Args を取り出す (SearchHelper)
// --------------------------------------------------

// (A) プライマリテンプレート（デフォルト）
// T が A<...> 以外の型（例: int, float, B）だった場合のフォールバック
template<typename T>
struct SearchHelper {
    // T が A でないので、Args が存在しない。
    // よって、U が何であれ「含まれない (false)」と判定する
    template<typename U>
    static constexpr bool contains = false;
};

// (B) 部分特殊化 (Partial Specialization)
// T が A<Args...> の形に一致した場合、こちらが使われる
template<typename... Args>
struct SearchHelper<A<Args...>> {
    // T から Args... を自動的に推論し、ヘルパー1 (is_in_pack) に渡す
    template<typename U>
    static constexpr bool contains = is_in_pack<U, Args...>;
};

// --------------------------------------------------
// 4. ユーザー定義の B (search() の実装)
// --------------------------------------------------
template<typename T>
class B {
public:
    template<typename U>
    void search() {
        // SearchHelper を通じて、T が A<...> かどうか、
        // もしそうなら U が Args... に含まれるかを一括で判定
        
        // `::template contains<U>` という書き方は、
        // contains がUに依存するテンプレート名であることをコンパイラに伝えるため
        constexpr bool found = SearchHelper<T>::template contains<U>;

        // if constexpr を使うことで、
        // `found` が false の場合、else 節のコードだけがコンパイルされ、
        // `found` が true の場合、if 節のコードだけがコンパイルされる
        if constexpr (found) {
            std::cout << "結果: 'U' は T の Args... の中に見つかりました。" << std::endl;
        } else {
            std::cout << "結果: 'U' は見つかりませんでした (TがAでないか、Argsに含まれていません)。" << std::endl;
        }
    }
};

// --------------------------------------------------
// 5. 実行例
// --------------------------------------------------
int main() {
    // 例1: T = A<int, double, char> の場合
    std::cout << "--- B<A<int, double, char>> のテスト ---" << std::endl;
    B<A<int, double, char>> b1;
    
    std::cout << "U = double を検索... ";
    b1.search<double>(); // 見つかる
    
    std::cout << "U = float を検索... ";
    b1.search<float>();  // 見つからない

    // 例2: T = A<float> の場合
    std::cout << "\n--- B<A<float>> のテスト ---" << std::endl;
    B<A<float>> b2;
    
    std::cout << "U = double を検索... ";
    b2.search<double>(); // 見つからない
    
    // 例3: T = int (A ではない型) の場合
    std::cout << "\n--- B<int> のテスト ---" << std::endl;
    B<int> b3;
    
    std::cout << "U = double を検索... ";
    b3.search<double>(); // 見つからない (SearchHelper の(A)が使われる)


    //素電化
    static constexpr double e = 1.602e-19; //[C]

    //電子質量
    static constexpr double m = 9.109e-31; //[kg]

    //平均電子密度
    static constexpr double Ne = 1e8;

    //光速度
    static constexpr double c = 3e8;

    //誘電率
    static constexpr double epsilon0 = 8.854e-12; //[F/m]

    //透磁率
    static constexpr double mu0 = 1./(epsilon0*c*c);

    //プラズマ周波数
    std::cout<< std::sqrt(Ne*e*e/(m*epsilon0));
    
    return 0;
}
