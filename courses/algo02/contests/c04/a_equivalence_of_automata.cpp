#include <iostream>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

namespace std {
template <>
struct hash<std::pair<size_t, size_t>> {
    std::size_t operator()(const std::pair<size_t, size_t>& pair) const {
        return std::hash<size_t>{}(pair.first) ^
               (std::hash<size_t>{}(pair.second) << 1);
    }
};
}  // namespace std

struct Node {
    size_t first;
    size_t second;
};

struct Graph {
    size_t size;
    std::vector<std::vector<size_t>> nodes;
    std::unordered_set<size_t> is_terminal;

    Graph(size_t size) : size(size), nodes(size) {}
};

bool AreEquivalent(Graph& first_graph, Graph& second_graph) {
    std::unordered_set<std::pair<size_t, size_t>> used;

    std::queue<Node> queue;
    queue.emplace(0, 0);

    while (!queue.empty()) {
        auto [first, second] = queue.front();
        queue.pop();

        if (first_graph.is_terminal.count(first) !=
            second_graph.is_terminal.count(second)) {
            return false;
        }

        used.insert({first, second});

        for (size_t i = 0;
             i < static_cast<size_t>(first_graph.nodes[first].size()); ++i) {
            size_t next_first = first_graph.nodes[first][i];

            size_t next_second = second_graph.nodes[second][i];

            if (used.count({next_first, next_second}) == 1) {
                continue;
            }

            queue.emplace(next_first, next_second);
        }
    }

    return true;
}

Graph InputGraph() {
    size_t states;
    size_t count_term_state;
    size_t alphabet_size;

    std::cin >> states >> count_term_state >> alphabet_size;

    Graph graph(states);

    for (size_t i = 0; i < count_term_state; ++i) {
        size_t term_node;
        std::cin >> term_node;

        graph.is_terminal.insert(term_node);
    }

    for (size_t i = 0; i < states * alphabet_size; ++i) {
        size_t stock;
        char symbol;
        size_t end;
        std::cin >> stock >> symbol >> end;

        graph.nodes[stock].push_back(end);
    }

    return graph;
}

int main() {
    Graph first_graph = InputGraph();
    Graph second_graph = InputGraph();

    if (AreEquivalent(first_graph, second_graph)) {
        std::cout << "EQUIVALENT\n";
    } else {
        std::cout << "NOT EQUIVALENT\n";
    }

    return 0;
}
