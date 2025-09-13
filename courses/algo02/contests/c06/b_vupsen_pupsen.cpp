#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <istream>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct Edge {
    int32_t from;
    int32_t to;

    bool operator==(const Edge& other) const = default;
};

template <>
struct std::hash<Edge> {
    size_t operator()(const Edge& edge) const {
        static std::hash<int32_t> func;
        static constexpr int32_t kOffset = 32;

        return (func(edge.from) << kOffset) ^ func(edge.to);
    }
};

struct EdgeProperties {
    int64_t flow;
    int64_t capacity;

    int64_t ResidualCap() const { return capacity - flow; }
};

class Graph {
public:
    Graph() {}
    Graph(int32_t count_vertex) : graph_(count_vertex) {}
    Graph(const Graph& other) : graph_(other.graph_) {}

    void AddEdge(int32_t from, int32_t to) {
        graph_[from].push_back(to);
        graph_[to].push_back(from);
    }

    std::vector<Edge> OutgoingEdges(int32_t vertex) const {
        std::vector<Edge> edges;

        edges.reserve(graph_[vertex].size());

        for (auto to : graph_[vertex]) {
            edges.push_back({vertex, to});
        }

        return edges;
    }

    const std::vector<int32_t>& OutGoingVertex(int32_t from) const {
        return graph_[from];
    }

    int32_t GetVertexCount() const { return graph_.size(); }

private:
    std::vector<std::vector<int32_t>> graph_;
};

template <typename Predicate, typename Iterator>
class FilteredIterator {
public:
    FilteredIterator(const Iterator& begin, const Iterator& end)
        : begin_(begin), kEnd(end) {}

    Iterator begin() const {  // NOLINT
        return begin_;
    }

    Iterator end() const {  // NOLINT
        return kEnd;
    }

private:
    Iterator begin_;
    const Iterator kEnd;
};

template <typename Predicate>
class FilteredGraph {
    class EdgeIterator {
    public:
        EdgeIterator(int32_t from,
                     const std::vector<int32_t>::const_iterator& to_begin,
                     const std::vector<int32_t>::const_iterator& to_end,
                     Predicate predicate)
            : from_(from),
              to_current_(to_begin),
              kToEnd(to_end),
              predicate_(predicate) {
            Next();
        }

        void Next() {
            while (to_current_ != kToEnd &&
                   !predicate_({from_, *to_current_})) {
                ++to_current_;
            }
        }

        bool operator!=(const EdgeIterator& other) const {
            return other.from_ != from_ || other.to_current_ != to_current_;
        }

        EdgeIterator operator++() {
            ++to_current_;
            Next();

            return EdgeIterator(from_, to_current_, kToEnd, predicate_);
        }

        Edge operator*() const { return {from_, *to_current_}; }

    private:
        int32_t from_;
        std::vector<int32_t>::const_iterator to_current_;
        const std::vector<int32_t>::const_iterator kToEnd;
        Predicate predicate_;
    };

public:
    FilteredGraph(std::shared_ptr<Graph> graph, Predicate predicate)
        : graph_(graph), predicate_(predicate) {}

    void AddEdge(int32_t from, int32_t to) { graph_->AddEdge(from, to); }

    int32_t GetCountVertex() const { return graph_->GetVertexCount(); }

    FilteredIterator<Predicate, EdgeIterator> OutgoingEdges(
        int32_t vertex) const {
        const auto& outgoing = graph_->OutGoingVertex(vertex);

        EdgeIterator begin(vertex, outgoing.begin(), outgoing.end(),
                           predicate_);

        EdgeIterator end(vertex, outgoing.end(), outgoing.end(), predicate_);

        return FilteredIterator<Predicate, EdgeIterator>(begin, end);
    }

private:
    std::shared_ptr<Graph> graph_;
    Predicate predicate_;
};

template <typename Vertex, typename Edge>
class BFSVisitor {
public:
    virtual void DiscoverVertex(Vertex /*vertex*/) {}
    virtual void ExamineVertex(Vertex /*vertex*/) {}
    virtual void ExamineEdge(const Edge& /*edge*/) {}
    virtual ~BFSVisitor() = default;
};

template <typename Vertex, typename Edge>
class DFSVisitor {
public:
    virtual void DiscoverVertex(Vertex /*vertex*/) {}
    virtual void ExamineEdge(const Edge& /*edge*/) {}
    virtual void FinishEdge(const Edge& /*edge*/) {}
    virtual ~DFSVisitor() = default;
};

class LevelBFSVisitor : public BFSVisitor<int32_t, Edge> {
public:
    LevelBFSVisitor(int32_t count_vertex, int32_t source, int32_t target)
        : source_(source), target_(target), level_(count_vertex) {}

    void DiscoverVertex(int32_t vertex) override {
        if (vertex == source_) {
            level_.assign(level_.size(), kNotFoundValue);
            level_[source_] = 0;
        }
    }

    void ExamineEdge(const Edge& edge) override {
        level_[edge.to] = level_[edge.from] + 1;
    }

    bool HasPathToTarget() const { return level_[target_] != kNotFoundValue; }

    bool IsFoundVertex(int32_t vertex) {
        return level_[vertex] != kNotFoundValue;
    }

    int32_t GetLevel(int32_t vertex) const { return level_[vertex]; }

private:
    int32_t source_;
    int32_t target_;
    const int32_t kNotFoundValue = -1;
    std::vector<int32_t> level_;
};

class IBuilderFlowNetwork;

class FlowNetwork {
    friend IBuilderFlowNetwork;

public:
    FlowNetwork() = default;
    void SetSource(int32_t source) { source = source; }

    void SetTarget(int32_t target) { target = target; }

    void SetGraph(std::shared_ptr<Graph> graph) { graph = graph; }

    void SetEdgeCapacity(const Edge& edge, int64_t new_capacity) {
        edges[edge].capacity = new_capacity;
    }

    void SetAllFlowZero() {
        for (auto& [edge, edge_properties] : edges) {
            edge_properties.flow = 0;
        }
    }

    template <typename Predicate>
    FilteredGraph<Predicate> ResidualNetworkView(Predicate predicate) {
        return FilteredGraph<Predicate>(graph, predicate);
    }

    int32_t GetSource() const { return source; }

    int32_t GetTarget() const { return target; }

    int32_t GetVertexCount() const { return graph->GetVertexCount(); }

    void ChangeEdgeFlow(const Edge& edge, int64_t delta) {
        edges[edge].flow += delta;
        edges[{edge.to, edge.from}].flow -= delta;
    }

    EdgeProperties& GetEdgeProperties(const Edge& edge) { return edges[edge]; }

private:
    int32_t source;
    int32_t target;
    std::shared_ptr<Graph> graph;
    std::unordered_map<Edge, EdgeProperties> edges;
};

class IBuilderFlowNetwork {
public:
    std::shared_ptr<FlowNetwork> GetFlowNetwork() { return flow_network_; }
    virtual void BuildFlowNetwork(
        int32_t /*source*/, int32_t /*target*/, int32_t /*count_vertex*/,
        const std::vector<Edge>& /*edges*/,
        const std::vector<int64_t>& /*edge_capacities*/) = 0;

    virtual void BuildGraph(int32_t /*count_vertex*/,
                            const std::vector<Edge>& /*edges*/) = 0;

    virtual void BuildEdgeCapacities(
        const std::vector<Edge>& /*edges*/,
        const std::vector<int64_t>& /*edge_capacities*/) = 0;

protected:
    std::shared_ptr<FlowNetwork> flow_network_;
};

class PushedDFSVisitor : public DFSVisitor<int32_t, Edge> {
public:
    PushedDFSVisitor(std::shared_ptr<FlowNetwork> flow_network)
        : pushed_(flow_network->GetVertexCount()),
          flow_network_(flow_network) {}

    void DiscoverVertex(int32_t vertex) override {
        if (vertex == flow_network_->GetSource()) {
            pushed_.assign(flow_network_->GetVertexCount(), 0);
        }
    }

    void ExamineEdge(const Edge& edge) override {
        EdgeProperties& edge_properties =
            flow_network_->GetEdgeProperties(edge);
        if (edge.from == flow_network_->GetSource()) {
            pushed_[edge.to] = edge_properties.ResidualCap();
            return;
        }
        pushed_[edge.to] =
            std::min(pushed_[edge.from], edge_properties.ResidualCap());
    }

    void FinishEdge(const Edge& edge) override {
        if (pushed_[flow_network_->GetTarget()] == 0) {
            return;
        }

        EdgeProperties& props = flow_network_->GetEdgeProperties(edge);
        pushed_[edge.from] = std::min({props.ResidualCap(), pushed_[edge.to]});

        flow_network_->ChangeEdgeFlow(edge, pushed_[edge.from]);
    }

    int32_t PushedFlow() const { return pushed_[flow_network_->GetSource()]; }

    bool ReachedTarget() const {
        return pushed_[flow_network_->GetTarget()] != 0;
    }

private:
    std::vector<int64_t> pushed_;
    std::shared_ptr<FlowNetwork> flow_network_;
};

class BuilderGoldIngotFlowNetwork : public IBuilderFlowNetwork {
public:
    void BuildFlowNetwork(
        int32_t source, int32_t target, int32_t count_vertex,
        const std::vector<Edge>& edges,
        const std::vector<int64_t>& edge_capacities) override {
        flow_network_ = std::make_shared<FlowNetwork>();

        flow_network_->SetSource(source);
        flow_network_->SetTarget(target);

        BuildGraph(count_vertex, edges);
        BuildEdgeCapacities(edges, edge_capacities);
    }

    void BuildGraph(int32_t count_vertex,
                    const std::vector<Edge>& edges) override {
        auto graph = std::make_shared<Graph>(count_vertex);

        for (const auto& edge : edges) {
            graph->AddEdge(edge.from, edge.to);
        }

        flow_network_->SetGraph(graph);
    }
    void BuildEdgeCapacities(
        const std::vector<Edge>& edges,
        const std::vector<int64_t>& edge_capacities) override {
        for (size_t i = 0; i < edges.size(); ++i) {
            flow_network_->SetEdgeCapacity(edges[i], edge_capacities[i]);
        }
    }
};

template <typename Visitor, typename Vertex, typename Graph>
void BFS(Vertex origin_vertex, const Graph& graph, Visitor& visitor) {
    std::unordered_set<Vertex> used;
    std::deque<Vertex> queue;

    visitor.DiscoverVertex(origin_vertex);
    queue.push_back(origin_vertex);
    used.insert(origin_vertex);

    while (!queue.empty()) {
        Vertex next_vertex = queue.front();
        queue.pop_front();
        visitor.ExamineVertex(next_vertex);

        for (const auto& edge : graph.OutgoingEdges(next_vertex)) {
            visitor.ExamineEdge(edge);
            Vertex target = edge.to;

            if (used.find(target) == used.end()) {
                visitor.DiscoverVertex(target);
                used.insert(target);
                queue.push_back(target);
            }
        }
    }
}

template <typename Visitor, typename Vertex, typename Graph>
void DFS(Vertex origin_vertex, const Graph& graph, Visitor& visitor) {
    std::unordered_set<Vertex> used;
    DFS(origin_vertex, graph, visitor, used);
}

template <typename Visitor, typename Vertex, typename Graph>
void DFS(Vertex origin_vertex, const Graph& graph, Visitor& visitor,
         std::unordered_set<Vertex>& used) {
    visitor.DiscoverVertex(origin_vertex);
    used.insert(origin_vertex);

    for (const auto& edge : graph.OutgoingEdges(origin_vertex)) {
        visitor.ExamineEdge(edge);

        Vertex target = edge.to;

        if (used.find(target) == used.end()) {
            used.insert(target);
            DFS(target, graph, visitor, used);
            visitor.FinishEdge(edge);
        }
    }
}

int32_t FindMaxFlow(std::shared_ptr<FlowNetwork> flow_network) {
    int32_t max_flow = 0;

    LevelBFSVisitor visitor(flow_network->GetVertexCount(),
                            flow_network->GetSource(),
                            flow_network->GetTarget());

    PushedDFSVisitor pushed_visitor(flow_network);

    auto filter_level_predicate = [&flow_network,
                                   &visitor](const Edge& edge) -> bool {
        EdgeProperties& edge_properties = flow_network->GetEdgeProperties(edge);

        return !visitor.IsFoundVertex(edge.to) &&
               edge_properties.ResidualCap() > 0;
    };

    auto filter_push_predicate = [&visitor, &pushed_visitor,
                                  &flow_network](const Edge& edge) -> bool {
        EdgeProperties& edge_properties = flow_network->GetEdgeProperties(edge);

        return !pushed_visitor.ReachedTarget() &&
               edge_properties.ResidualCap() > 0 &&
               visitor.GetLevel(edge.to) == visitor.GetLevel(edge.from) + 1;
    };

    auto graph = flow_network->ResidualNetworkView(filter_level_predicate);
    auto level_graph = flow_network->ResidualNetworkView(filter_push_predicate);

    do {
        BFS(flow_network->GetSource(), graph, visitor);
        int32_t pushed = 0;
        do {
            DFS(flow_network->GetSource(), level_graph, pushed_visitor);
            pushed = pushed_visitor.PushedFlow();
            max_flow += pushed;
        } while (pushed > 0);

    } while (visitor.HasPathToTarget());

    return max_flow;
}

template <typename Predicate>
int32_t BinarySearch(Predicate condition, int32_t left_bound,
                     int32_t right_bound) {
    while (left_bound + 1 != right_bound) {
        int32_t median = left_bound + (right_bound - left_bound) / 2;
        if (condition(median)) {
            right_bound = median;
        } else {
            left_bound = median;
        }
    }
    return right_bound;
}

struct FriendPair {
    int32_t vupsen;
    int32_t pupsen;
};

struct Grub {
    bool is_pupsen;
    int32_t kill_cost;
};

void SetMaxTargetCapacity(std::shared_ptr<FlowNetwork> flow_network,
                          int32_t target_capacity) {
    for (int32_t node = flow_network->GetSource() + 1;
         node != flow_network->GetTarget(); ++node) {
        flow_network->SetEdgeCapacity({node, flow_network->GetTarget()},
                                      target_capacity);
    }
}

std::shared_ptr<FlowNetwork> BuildFlowNetwork(
    const std::vector<Grub>& kill_costs,
    const std::vector<FriendPair>& friend_pairs) {
    BuilderGoldIngotFlowNetwork builder;

    int32_t count_vertex = kill_costs.size() + 1;
    int32_t source = 0;
    int32_t target = count_vertex - 1;
    int32_t max_kill_cost = 0;

    std::vector<Edge> edges;
    std::vector<int64_t> edge_capacities;

    for (int32_t node = 1; node != target; ++node) {
        if (!kill_costs[node].is_pupsen) {
            edges.push_back({source, node});
            edge_capacities.push_back(kill_costs[node].kill_cost);
        } else {
            edges.push_back({node, target});
            edge_capacities.push_back(kill_costs[node].kill_cost);
        }

        max_kill_cost = std::max(max_kill_cost, kill_costs[node].kill_cost);
    }

    for (const auto& [vupsen, pupsen] : friend_pairs) {
        edges.push_back({vupsen, pupsen});
        edge_capacities.push_back(max_kill_cost);
    }

    builder.BuildFlowNetwork(source, target, count_vertex, edges,
                             edge_capacities);
    return builder.GetFlowNetwork();
}

int32_t FindMinKillCost(const std::vector<Grub>& kill_costs,
                        const std::vector<FriendPair>& friend_pairs) {
    auto flow_network = BuildFlowNetwork(kill_costs, friend_pairs);

    return FindMaxFlow(flow_network);
}

struct InputData {
    std::vector<Grub> kill_costs;
    std::vector<FriendPair> friend_pairs;

    InputData(int32_t count_people) : kill_costs(count_people + 1) {}
};

InputData Input(std::istream& input = std::cin) {
    int32_t count_people;
    int32_t count_pair;
    input >> count_people >> count_pair;

    InputData input_data(count_people);

    for (int32_t i = 1; i <= count_people; ++i) {
        int32_t is_pupsen;
        int32_t kill_cost;
        input >> is_pupsen >> kill_cost;
        input_data.kill_costs[i] = Grub(is_pupsen == 1, kill_cost);
    }

    for (int32_t i = 0; i < count_pair; ++i) {
        int32_t from;
        int32_t to;
        input >> from >> to;

        if (input_data.kill_costs[from].is_pupsen ==
            input_data.kill_costs[to].is_pupsen) {
            continue;
        }

        if (input_data.kill_costs[from].is_pupsen) {
            std::swap(from, to);
        }

        input_data.friend_pairs.emplace_back(from, to);
    }

    return input_data;
}

void Output(int32_t kill_cost, std::ostream& output = std::cout) {
    output << kill_cost << std::endl;
}

int main() {
    InputData input_data = Input(std::cin);
    std::vector<Grub> kill_costs = std::move(input_data.kill_costs);
    std::vector<FriendPair> friend_pairs = std::move(input_data.friend_pairs);

    Output(FindMinKillCost(kill_costs, friend_pairs));

    return 0;
}
