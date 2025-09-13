#include <climits>
#include <iostream>
#include <memory>
#include <queue>
#include <vector>

constexpr int64_t kInf = LLONG_MAX;

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
    int to;
    int rev;
    int64_t cap;
};

struct Graph {
    int n;
    std::vector<std::vector<Edge>> adj;
    std::vector<int64_t> level;
    std::vector<size_t> stop;

    explicit Graph(int nodes)
        : n(nodes), adj(nodes), level(nodes), stop(nodes) {}
};

inline void AddEdge(Graph& gr, int from, int to, int64_t cap) {
    gr.adj[from].push_back({to, static_cast<int>(gr.adj[to].size()), cap});
    gr.adj[to].push_back({from, static_cast<int>(gr.adj[from].size()) - 1, 0});
}

class IBuilderNetworkFlow;

class NetworkFlow {
    friend class IBuilderNetworkFlow;

public:
    explicit NetworkFlow(int nodes) : gr_(nodes) {}

    const Graph& GetBaseGraph() const { return gr_; }

    void AddEdge(int from, int to, int64_t cap) {
        gr_.adj[from].push_back(
            {to, static_cast<int>(gr_.adj[to].size()), cap});
        gr_.adj[to].push_back(
            {from, static_cast<int>(gr_.adj[from].size()) - 1, 0});
    }

private:
    Graph gr_;
};

class IBuilderNetworkFlow {
public:
    std::shared_ptr<NetworkFlow> GetNetworkFlow() { return net_; }

protected:
    std::shared_ptr<NetworkFlow> net_;
};

class BuilderGoldNetworkFlow : public IBuilderNetworkFlow {
public:
    void BuildGraph(const std::vector<int64_t>& golds,
                    const std::vector<std::vector<int64_t>>& trust) {
        int num_people = static_cast<int>(golds.size()) - 1;
        int source = 0;
        int sink = num_people + 1;

        net_ = std::make_shared<NetworkFlow>(sink + 1);

        for (int i = 1; i <= num_people; ++i) {
            net_->AddEdge(source, i, golds[i]);
        }

        for (int from = 1; from <= num_people; ++from) {
            for (int to : trust[from]) {
                net_->AddEdge(from, to, kInf);
            }
        }
    }
};

bool BFS(int source, int sink, Graph& gr) {
    std::fill(gr.level.begin(), gr.level.end(), -1);
    gr.level[source] = 0;

    std::queue<int> que;
    que.push(source);

    while (!que.empty() && gr.level[sink] == -1) {
        int next = que.front();
        que.pop();

        for (const Edge& edge : gr.adj[next]) {
            if (edge.cap > 0 && gr.level[edge.to] == -1) {
                gr.level[edge.to] = gr.level[next] + 1;
                que.push(edge.to);
            }
        }
    }
    return gr.level[sink] != -1;
}

int64_t DFS(int vert, Graph& gr, int sink, int64_t flow) {
    if (vert == sink || flow == 0) {
        return flow;
    }
    for (size_t& cid = gr.stop[vert]; cid < gr.adj[vert].size(); ++cid) {
        Edge& edge = gr.adj[vert][cid];
        if (gr.level[edge.to] != gr.level[vert] + 1 || edge.cap <= 0) {
            continue;
        }
        int64_t pushed = DFS(edge.to, gr, sink, std::min(flow, edge.cap));
        if (pushed > 0) {
            edge.cap -= pushed;
            gr.adj[edge.to][edge.rev].cap += pushed;
            return pushed;
        }
    }
    return 0;
}

int64_t FindMaxFlow(int source, int sink, Graph& gr) {
    int64_t max_flow = 0;
    while (BFS(source, sink, gr)) {
        gr.stop.assign(gr.n, 0);
        while (int64_t pushed = DFS(source, gr, sink, kInf)) {
            max_flow += pushed;
        }
    }
    return max_flow;
}

int64_t FindGoldOfBusiestPerson(std::shared_ptr<NetworkFlow>& net,
                                int64_t total_gold, int64_t max_gold) {
    Graph base_graph = net->GetBaseGraph();
    int source = 0;
    int sink = base_graph.n - 1;

    auto predicate = [&](int64_t gold_bound) -> bool {
        Graph gr = base_graph;

        for (int source = 1; source < sink; ++source) {
            AddEdge(gr, source, sink, gold_bound);
        }

        return FindMaxFlow(source, sink, gr) == total_gold;
    };

    return BinarySearch(0, static_cast<size_t>(max_gold), predicate);
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    BuilderGoldNetworkFlow builder;

    int64_t peop_size;
    int64_t pair_size;
    std::cin >> peop_size >> pair_size;

    int64_t total_gold = 0;
    int64_t max_gold = 0;
    std::vector<int64_t> golds(peop_size + 1);
    for (int64_t i = 1; i <= peop_size; ++i) {
        std::cin >> golds[i];
        total_gold += golds[i];
        max_gold = std::max(max_gold, golds[i]);
    }

    std::vector<std::vector<int64_t>> trust(peop_size + 1);
    for (int64_t i = 0; i < pair_size; ++i) {
        int64_t from;
        int64_t to;
        std::cin >> from >> to;
        trust[from].push_back(to);
    }

    builder.BuildGraph(golds, trust);
    auto net = builder.GetNetworkFlow();

    std::cout << FindGoldOfBusiestPerson(net, total_gold, max_gold) << '\n';

    return 0;
}
