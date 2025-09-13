#include <climits>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

using VertexId = int32_t;
using EdgeId = int32_t;
using FlowUnits = int64_t;

constexpr FlowUnits kInfFlow = std::numeric_limits<FlowUnits>::max();

/**
 * Finds the minimal int t ∈ (left, right] s.t. predicate(t) == true.
 *
 * Requirements:
 * - predicate(m) == false for all m <= left.
 * - predicate(m) == true for all m >= right.
 * - predicate(x) == true -> predicate(y) == true for all y > x.
 *
 * Returns:
 * - The smallest int t for which predicate(t) == true.
 * - If no such t exists, returns right.
 */
template <typename Predicate>
int64_t BinarySearch(int64_t left, int64_t right, Predicate predicate) {
    while (left + 1 != right) {
        int64_t middle = left + (right - left) / 2;
        if (predicate(middle)) {
            right = middle;
        } else {
            left = middle;
        }
    }

    return right;
}

struct Edge {
    VertexId to;
    EdgeId reverse;
    FlowUnits capacity;
};

struct FlowNetwork {
    int32_t capacity;
    std::vector<std::vector<Edge>> adjacent;

    explicit FlowNetwork(int32_t nodes) : capacity(nodes), adjacent(nodes) {}

    void AddEdge(VertexId from, VertexId to, int64_t capacity) {
        adjacent[from].push_back(
            {to, static_cast<EdgeId>(adjacent[to].size()), capacity});
        adjacent[to].push_back(
            {from, static_cast<EdgeId>(adjacent[from].size()) - 1, 0});
    }
};

class DinicAlgorithm {
private:
    std::vector<int64_t> level_;
    std::vector<size_t> stop_;

    bool BFS(VertexId source, VertexId sink, FlowNetwork& network) {
        std::fill(level_.begin(), level_.end(), -1);
        level_[source] = 0;

        std::queue<VertexId> queue;
        queue.push(source);

        while (!queue.empty() && level_[sink] == -1) {
            VertexId next = queue.front();
            queue.pop();

            for (const Edge& edge : network.adjacent[next]) {
                if (edge.capacity > 0 && level_[edge.to] == -1) {
                    level_[edge.to] = level_[next] + 1;
                    queue.push(edge.to);
                }
            }
        }
        return level_[sink] != -1;
    }

    int64_t DFS(VertexId vertex, FlowNetwork& network, VertexId sink,
                int64_t flow) {
        if (vertex == sink || flow == 0) {
            return flow;
        }

        for (size_t& cid = stop_[vertex]; cid < network.adjacent[vertex].size();
             ++cid) {
            Edge& edge = network.adjacent[vertex][cid];

            if (level_[edge.to] != level_[vertex] + 1 || edge.capacity <= 0) {
                continue;
            }

            int64_t pushed =
                DFS(edge.to, network, sink, std::min(flow, edge.capacity));

            if (pushed > 0) {
                edge.capacity -= pushed;
                network.adjacent[edge.to][edge.reverse].capacity += pushed;
                return pushed;
            }
        }
        return 0;
    }

public:
    explicit DinicAlgorithm(int32_t nodes) : level_(nodes), stop_(nodes) {}

    int64_t ComputeMaxFlow(FlowNetwork& network, VertexId source,
                           VertexId sink) {
        int64_t max_flow = 0;

        while (BFS(source, sink, network)) {
            stop_.assign(network.capacity, 0);

            while (int64_t pushed = DFS(source, network, sink, kInfFlow)) {
                max_flow += pushed;
            }
        }

        return max_flow;
    }
};

FlowNetwork BuildGoldFlowNetwork(
    const std::vector<FlowUnits>& golds,
    const std::vector<std::vector<VertexId>>& trust) {
    int32_t num_people = static_cast<int32_t>(golds.size()) - 1;
    VertexId source = 0;
    VertexId sink = num_people + 1;

    FlowNetwork network = FlowNetwork(sink + 1);

    for (VertexId i = 1; i <= num_people; ++i) {
        network.AddEdge(source, i, golds[i]);
    }

    for (VertexId from = 1; from <= num_people; ++from) {
        for (VertexId to : trust[from]) {
            network.AddEdge(from, to, kInfFlow);
        }
    }

    return network;
}

int64_t FindGoldOfBusiestPerson(FlowNetwork& network, FlowUnits total_gold,
                                FlowUnits max_gold) {
    VertexId source = 0;
    VertexId sink = network.capacity - 1;

    auto predicate = [&](int64_t gold_bound) -> bool {
        FlowNetwork network_current = network;

        for (VertexId source = 1; source < sink; ++source) {
            network_current.AddEdge(source, sink, gold_bound);
        }

        DinicAlgorithm dinic(network_current.capacity);
        return dinic.ComputeMaxFlow(network_current, source, sink) ==
               total_gold;
    };

    return BinarySearch(0, max_gold, predicate);
}

struct GoldTrustConfig {
    std::vector<FlowUnits> golds;
    std::vector<std::vector<VertexId>> trust;
    FlowUnits total_gold;
    FlowUnits max_gold;

    GoldTrustConfig(int64_t people_size)
        : golds(people_size + 1),
          trust(people_size + 1),
          total_gold(0),
          max_gold(0) {}
};

GoldTrustConfig ReadGoldTrustInput(std::istream& input = std::cin) {
    int64_t people_size;
    int64_t pair_size;
    input >> people_size >> pair_size;

    GoldTrustConfig config(people_size);

    for (int64_t i = 1; i <= people_size; ++i) {
        input >> config.golds[i];

        config.total_gold += config.golds[i];
        config.max_gold = std::max(config.max_gold, config.golds[i]);
    }

    for (int64_t i = 0; i < pair_size; ++i) {
        VertexId from;
        VertexId to;
        input >> from >> to;

        config.trust[from].push_back(to);
    }

    return config;
}

int main() {
    GoldTrustConfig config = ReadGoldTrustInput();

    FlowNetwork network = BuildGoldFlowNetwork(config.golds, config.trust);

    int64_t gold_of_busiest_person =
        FindGoldOfBusiestPerson(network, config.total_gold, config.max_gold);
    std::cout << gold_of_busiest_person << '\n';

    return 0;
}
