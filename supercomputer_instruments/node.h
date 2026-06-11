#ifndef NODE_H
#define NODE_H
#include <optional>
#include <vector>
#include <utility>

enum class Hand { LEFT, RIGHT };

struct Endpoint {
    int node;
    Hand hand;
};

struct Node {
    std::optional<Endpoint> left;
    std::optional<Endpoint> right;
};

// 戻り値:
//   nullopt            -> 接続されていない
//   pair(aHand, bHand) -> 接続されている手
std::optional<std::pair<Hand, Hand>>
inline get_connection_hands(const std::vector<Node>& nodes,
                     int a, int b)
{
    // a の left
    if (nodes[a].left && nodes[a].left->node == b) {
        Hand aHand = Hand::LEFT;
        Hand bHand = nodes[a].left->hand;
        return std::make_pair(aHand, bHand);
    }

    // a の right
    if (nodes[a].right && nodes[a].right->node == b) {
        Hand aHand = Hand::RIGHT;
        Hand bHand = nodes[a].right->hand;
        return std::make_pair(aHand, bHand);
    }

    // 接続されていない
    return std::nullopt;
}
#endif //NODE_H
