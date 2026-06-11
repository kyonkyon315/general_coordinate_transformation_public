#ifndef COMM_GRAPH_CONSTRUCTOR
#define COMM_GRAPH_CONSTRUCTOR

#include <vector>
#include <utility>
#include <algorithm>

#include "node.h"


inline Hand opposite(Hand h) {
    return (h == Hand::LEFT ? Hand::RIGHT : Hand::LEFT);
}

inline bool has_pocket(const Node& n) {
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

inline std::pair<std::vector<Ring>, std::vector<Linear>>
buildRingsAndLinears(std::vector<Node> nodes)
{
    int N = nodes.size();
    for(int i=0;i<N;i++){
        if(i == nodes[i].left.value().node){
            nodes[i].left.reset();
        }
        if(i == nodes[i].right.value().node){
            nodes[i].right.reset();
        }
    }

    std::vector<bool> visited(N, false);

    std::vector<Ring> rings;
    std::vector<Linear> linears;

    // =========================
    // 1. Linear をポッケ端から構築
    // =========================
    for (int i = 0; i < N; ++i) {
        if (visited[i]) continue;
        if (!has_pocket(nodes[i])) continue;

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

#endif //COMM_GRAPH_CONSTRUCTOR
