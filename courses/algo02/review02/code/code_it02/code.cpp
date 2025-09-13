#include <climits>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
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
    EdgeId rev;
    FlowUnits cap;
};

struct FlowNetwork {
    int32_t capacity;
    std::vector<std::vector<Edge>> adj;
    std::vector<int64_t> level;
    std::vector<size_t> stop;

    explicit FlowNetwork(int32_t nodes)
        : capacity(nodes), adj(nodes), level(nodes), stop(nodes) {}

    void AddEdge(VertexId from, VertexId to, int64_t cap) {
        adj[from].push_back({to, static_cast<EdgeId>(adj[to].size()), cap});
        adj[to].push_back({from, static_cast<EdgeId>(adj[from].size()) - 1, 0});
    }
};

std::shared_ptr<FlowNetwork> BuildGoldFlowNetwork(
    const std::vector<FlowUnits>& golds,
    const std::vector<std::vector<VertexId>>& trust) {
    int32_t num_people = static_cast<int32_t>(golds.size()) - 1;
    VertexId source = 0;
    VertexId sink = num_people + 1;

    auto net = std::make_shared<FlowNetwork>(sink + 1);

    for (VertexId i = 1; i <= num_people; ++i) {
        net->AddEdge(source, i, golds[i]);
    }

    for (VertexId from = 1; from <= num_people; ++from) {
        for (VertexId to : trust[from]) {
            net->AddEdge(from, to, kInfFlow);
        }
    }

    return net;
}

bool BFS(VertexId source, VertexId sink, FlowNetwork& net) {
    std::fill(net.level.begin(), net.level.end(), -1);
    net.level[source] = 0;

    std::queue<VertexId> que;
    que.push(source);

    while (!que.empty() && net.level[sink] == -1) {
        VertexId next = que.front();
        que.pop();

        for (const Edge& edge : net.adj[next]) {
            if (edge.cap > 0 && net.level[edge.to] == -1) {
                net.level[edge.to] = net.level[next] + 1;
                que.push(edge.to);
            }
        }
    }
    return net.level[sink] != -1;
}

int64_t DFS(VertexId vert, FlowNetwork& net, VertexId sink, int64_t flow) {
    if (vert == sink || flow == 0) {
        return flow;
    }

    for (size_t& cid = net.stop[vert]; cid < net.adj[vert].size(); ++cid) {
        Edge& edge = net.adj[vert][cid];

        if (net.level[edge.to] != net.level[vert] + 1 || edge.cap <= 0) {
            continue;
        }

        int64_t pushed = DFS(edge.to, net, sink, std::min(flow, edge.cap));

        if (pushed > 0) {
            edge.cap -= pushed;
            net.adj[edge.to][edge.rev].cap += pushed;
            return pushed;
        }
    }
    return 0;
}

int64_t FindMaxFlow(VertexId source, VertexId sink, FlowNetwork& net) {
    int64_t max_flow = 0;

    while (BFS(source, sink, net)) {
        net.stop.assign(net.capacity, 0);

        while (int64_t pushed = DFS(source, net, sink, kInfFlow)) {
            max_flow += pushed;
        }
    }

    return max_flow;
}

int64_t FindGoldOfBusiestPerson(std::shared_ptr<FlowNetwork> net,
                                FlowUnits total_gold, FlowUnits max_gold) {
    VertexId source = 0;
    VertexId sink = net->capacity - 1;

    auto predicate = [&](int64_t gold_bound) -> bool {
        FlowNetwork network = *net;

        for (VertexId source = 1; source < sink; ++source) {
            network.AddEdge(source, sink, gold_bound);
        }

        return FindMaxFlow(source, sink, network) == total_gold;
    };

    return BinarySearch(0, max_gold, predicate);
}

void Input(std::vector<FlowUnits>& golds,
           std::vector<std::vector<VertexId>>& trust, FlowUnits& total_gold,
           FlowUnits& max_gold) {
    int64_t peop_size;
    int64_t pair_size;
    std::cin >> peop_size >> pair_size;

    golds.resize((peop_size + 1));
    for (int64_t i = 1; i <= peop_size; ++i) {
        std::cin >> golds[i];
        total_gold += golds[i];
        max_gold = std::max(max_gold, golds[i]);
    }

    trust.resize(peop_size + 1);
    for (int64_t i = 0; i < pair_size; ++i) {
        VertexId from;
        VertexId to;
        std::cin >> from >> to;
        trust[from].push_back(to);
    }
}

int main() {
    std::vector<FlowUnits> golds;
    std::vector<std::vector<VertexId>> trust;
    FlowUnits total_gold = 0;
    FlowUnits max_gold = 0;

    Input(golds, trust, total_gold, max_gold);

    std::shared_ptr<FlowNetwork> net = BuildGoldFlowNetwork(golds, trust);

    std::cout << FindGoldOfBusiestPerson(net, total_gold, max_gold) << '\n';

    return 0;
}
