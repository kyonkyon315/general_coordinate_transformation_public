#include <vector>
#include <optional>
#include <utility>
#include <algorithm>

enum class Hand { LEFT, RIGHT };

struct Endpoint {
    int node;
    Hand hand;
};

struct Node {
    std::optional<Endpoint> left;
    std::optional<Endpoint> right;
};

Hand opposite(Hand h) {
    return (h == Hand::LEFT ? Hand::RIGHT : Hand::LEFT);
}

bool hasPocket(const Node& n) {
    return !n.left.has_value() || !n.right.has_value();
}

class Ring {
public:
    std::vector<int> nodes;

    explicit Ring(std::vector<int> n)
        : nodes(std::move(n)) {}

    void normalize() {
        int n = nodes.size();
        if (n == 0) return;
        if (n == 1) return; // 単一ノードリング（特殊）

        // --- 1. 最小値の位置を探す ---
        int minVal = nodes[0];
        int minIdx = 0;
        for (int i = 1; i < n; ++i) {
            if (nodes[i] < minVal) {
                minVal = nodes[i];
                minIdx = i;
            }
        }

        // 隣接2ノード（リングなので mod）
        int prev = nodes[(minIdx - 1 + n) % n];
        int next = nodes[(minIdx + 1) % n];

        // --- 2. 次に来るのは小さい方 ---
        bool forward;
        if (next < prev) {
            // forward: min -> next
            forward = true;
        } else {
            // backward: min -> prev
            forward = false;
        }

        // --- 3. 回転＋向きの適用 ---
        std::vector<int> result;
        result.reserve(n);
        result.push_back(minVal);

        int idx = minIdx;
        for (int k = 1; k < n; ++k) {
            if (forward) {
                idx = (idx + 1) % n;
            } else {
                idx = (idx - 1 + n) % n;
            }
            result.push_back(nodes[idx]);
        }

        nodes = std::move(result);
    }
};

class Linear {
public:
    std::vector<int> nodes;

    explicit Linear(std::vector<int> n)
        : nodes(std::move(n)) {}

    
    void normalize() {
        if (nodes.empty()) return;

        int a = nodes.front();
        int b = nodes.back();

        // 小さい端点が先頭になるように
        if (b < a) {
            std::reverse(nodes.begin(), nodes.end());
        }
    }
};

std::pair<std::vector<Ring>, std::vector<Linear>>
buildRingsAndLinears(const std::vector<Node>& nodes)
{
    int N = nodes.size();
    std::vector<bool> visited(N, false);

    std::vector<Ring> rings;
    std::vector<Linear> linears;

    // =========================
    // 1. Linear をポッケ端から構築
    // =========================
    for (int i = 0; i < N; ++i) {
        if (visited[i]) continue;
        if (!hasPocket(nodes[i])) continue;

        std::vector<int> chain;

        int cur = i;

        // 両手ポッケ → 単独 Linear
        if (!nodes[cur].left && !nodes[cur].right) {
            visited[cur] = true;
            chain.push_back(cur);
            linears.emplace_back(chain);
            continue;
        }

        // 出ていく手（接続されている側）
        Hand outHand;
        if (nodes[cur].left)  outHand = Hand::LEFT;
        else                  outHand = Hand::RIGHT;

        while (true) {
            if (visited[cur]) break;
            visited[cur] = true;
            chain.push_back(cur);

            auto &ep = (outHand == Hand::LEFT ?
                        nodes[cur].left : nodes[cur].right);

            if (!ep) break;

            int next = ep->node;
            Hand nextIn = ep->hand;

            if (visited[next]) break;

            cur = next;
            outHand = opposite(nextIn);

            auto &nextEp = (outHand == Hand::LEFT ?
                            nodes[cur].left : nodes[cur].right);

            if (!nextEp) {
                if (!visited[cur]) {
                    visited[cur] = true;
                    chain.push_back(cur);
                }
                break;
            }
        }

        linears.emplace_back(chain);
    }

    // =========================
    // 2. 残りは Ring
    // =========================
    for (int i = 0; i < N; ++i) {
        if (visited[i]) continue;

        std::vector<int> ring;

        int start = i;
        int cur = i;
        Hand outHand = Hand::LEFT;

        visited[cur] = true;
        ring.push_back(cur);

        while (true) {
            auto &ep = (outHand == Hand::LEFT ?
                        nodes[cur].left : nodes[cur].right);

            if (!ep) break; // 異常系（Ringのはず）

            int next = ep->node;
            Hand nextIn = ep->hand;

            if (next == start) {
                break; // 正常に閉じた
            }

            if (visited[next]) {
                break; // 念のため
            }

            cur = next;
            visited[cur] = true;
            ring.push_back(cur);

            outHand = opposite(nextIn);
        }

        rings.emplace_back(ring);
    }
    for(int i=0;i<rings.size();i++){
        rings[i].normalize();
    }
    for(int i=0;i<linears.size();i++){
        linears[i].normalize();
    }

    return {rings, linears};
}

#include <iostream>

int main()
{
    int N = 6;
    std::vector<Node> nodes(N);

    // --- Ring: 0-1-2 ---
    nodes[0].right = Endpoint{1, Hand::LEFT};
    nodes[1].left  = Endpoint{0, Hand::RIGHT};

    nodes[1].right = Endpoint{2, Hand::LEFT};
    nodes[2].left  = Endpoint{1, Hand::RIGHT};

    nodes[2].right = Endpoint{0, Hand::LEFT};
    nodes[0].left  = Endpoint{2, Hand::RIGHT};

    // --- Linear: 3-4 ---
    nodes[3].right = Endpoint{4, Hand::LEFT};
    nodes[4].left  = Endpoint{3, Hand::RIGHT};

    // --- Single: 5 ---
    // nodes[5] は両手ポッケ

    auto result = buildRingsAndLinears(nodes);

    auto& rings   = result.first;
    auto& linears = result.second;

    std::cout << "=== Rings ===\n";
    for (auto& r : rings) {
        std::cout << "Ring: ";
        for (int n : r.nodes) std::cout << n << " ";
        std::cout << "\n";
    }

    std::cout << "=== Linears ===\n";
    for (auto& l : linears) {
        std::cout << "Linear: ";
        for (int n : l.nodes) std::cout << n << " ";
        std::cout << "\n";
    }

    return 0;
}