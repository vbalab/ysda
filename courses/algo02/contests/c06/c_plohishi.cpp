#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <istream>
#include <memory>
#include <ostream>
#include <string>
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

class LevelBFShVisitor : public BFSVisitor<int, Edge> {
public:
    LevelBFShVisitor(int count_vertex, int source, int target)
        : source_(source), target_(target), level_(count_vertex) {}

    void DiscoverVertex(int vertex) override {
        if (vertex == source_) {
            level_.assign(level_.size(), kNotFoundValue);
            level_[source_] = 0;
        }
    }

    void ExamineEdge(const Edge& edge) override {
        level_[edge.to] = level_[edge.from] + 1;
    }

    bool HasPathToTarget() const { return level_[target_] != kNotFoundValue; }

    bool IsFoundVertex(int vertex) { return level_[vertex] != kNotFoundValue; }

    int GetLevel(int vertex) const { return level_[vertex]; }

private:
    int source_;
    int target_;
    const int kNotFoundValue = -1;
    std::vector<int> level_;
};

class IBuilderFlowNetwork;

struct Name {
    int first_word;
    int second_word;

    Name(int first_word, int second_word)
        : first_word(first_word), second_word(second_word) {}
};

class FlowNetwork {
    friend IBuilderFlowNetwork;

public:
    FlowNetwork() = default;
    void SetSource(int source) { source_ = source; }

    void SetTarget(int target) { target_ = target; }

    void SetGraph(std::shared_ptr<Graph> graph) { graph_ = graph; }

    void SetEdgeCapacity(const Edge& edge, int64_t new_capacity) {
        edges_[edge].capacity = new_capacity;
    }

    void SetAllFlowZero() {
        for (auto& [edge, edge_properties] : edges_) {
            edge_properties.flow = 0;
        }
    }

    template <typename Predicate>
    FilteredGraph<Predicate> ResidualNetworkView(Predicate predicate) {
        return FilteredGraph<Predicate>(graph_, predicate);
    }

    int GetSource() const { return source_; }

    int GetTarget() const { return target_; }

    int GetVertexCount() const { return graph_->GetVertexCount(); }

    void ChangeEdgeFlow(const Edge& edge, int64_t delta) {
        edges_[edge].flow += delta;
        edges_[{edge.to, edge.from}].flow -= delta;
    }

    EdgeProperties& GetEdgeProperties(const Edge& edge) { return edges_[edge]; }

    int GetUniqueCount(const std::vector<bool>& is_first_part) {
        const int kSize = static_cast<int>(is_first_part.size());
        int unque_count = 0;

        for (int node = 1; node < kSize; ++node) {
            if (is_first_part[node]) {
                if (edges_[{source_, node}].ResidualCap() != 0) {
                    ++unque_count;
                }
            } else {
                if (edges_[{node, target_}].ResidualCap() != 0) {
                    ++unque_count;
                }
            }
        }

        return unque_count;
    }

private:
    int source_;
    int target_;

    std::shared_ptr<Graph> graph_;
    std::unordered_map<Edge, EdgeProperties> edges_;
};

class IBuilderFlowNetwork {
public:
    std::shared_ptr<FlowNetwork> GetFlowNetwork() { return net_; }

    virtual void BuildFlowNetwork(
        int /*source*/, int /*target*/, int /*count_vertex*/,
        const std::vector<Edge>& /*edges*/,
        const std::vector<int64_t>& /*edge_capacities*/) = 0;

    virtual void BuildGraph(int /*count_vertex*/,
                            const std::vector<Edge>& /*edges*/) = 0;

    virtual void BuildEdgeCapacities(
        const std::vector<Edge>& /*edges*/,
        const std::vector<int64_t>& /*edge_capacities*/) = 0;

protected:
    std::shared_ptr<FlowNetwork> net_;
};

class PushedDFSVisitor : public DFSVisitor<int, Edge> {
public:
    PushedDFSVisitor(std::shared_ptr<FlowNetwork> net)
        : pushed_(net->GetVertexCount()), net_(net) {}

    void DiscoverVertex(int vertex) override {
        if (vertex == net_->GetSource()) {
            pushed_.assign(net_->GetVertexCount(), 0);
        }
    }

    void ExamineEdge(const Edge& edge) override {
        EdgeProperties& edge_properties = net_->GetEdgeProperties(edge);
        if (edge.from == net_->GetSource()) {
            pushed_[edge.to] = edge_properties.ResidualCap();
            return;
        }
        pushed_[edge.to] =
            std::min(pushed_[edge.from], edge_properties.ResidualCap());
    }

    void FinishEdge(const Edge& edge) override {
        if (pushed_[net_->GetTarget()] == 0) {
            return;
        }
        EdgeProperties& edge_properties = net_->GetEdgeProperties(edge);
        pushed_[edge.from] =
            std::min({edge_properties.ResidualCap(), pushed_[edge.to]});

        net_->ChangeEdgeFlow(edge, pushed_[edge.from]);
    }

    int PushedFlow() const { return pushed_[net_->GetSource()]; }

    bool ReachedTarget() const { return pushed_[net_->GetTarget()] != 0; }

private:
    std::vector<int64_t> pushed_;
    std::shared_ptr<FlowNetwork> net_;
};

template <typename GraphType>
auto OutgoingEdges(const GraphType& graph, int vertex)
    -> decltype(graph.OutgoingEdges(vertex)) {
    return graph.OutgoingEdges(vertex);
}

int GetTarget(const Edge& edge) { return edge.to; }

class BuilderGoldIngotFlowNetwork : public IBuilderFlowNetwork {
public:
    void BuildFlowNetwork(
        int source, int target, int count_vertex,
        const std::vector<Edge>& edges,
        const std::vector<int64_t>& edge_capacities) override {
        net_ = std::make_shared<FlowNetwork>();

        net_->SetSource(source);
        net_->SetTarget(target);

        BuildGraph(count_vertex, edges);
        BuildEdgeCapacities(edges, edge_capacities);
    }

    void BuildGraph(int count_vertex, const std::vector<Edge>& edges) override {
        auto graph = std::make_shared<Graph>(count_vertex);

        for (const auto& edge : edges) {
            graph->AddEdge(edge.from, edge.to);
        }

        net_->SetGraph(graph);
    }

    void BuildEdgeCapacities(
        const std::vector<Edge>& edges,
        const std::vector<int64_t>& edge_capacities) override {
        for (int edge = 0; edge < static_cast<int>(edges.size()); ++edge) {
            net_->SetEdgeCapacity(edges[edge], edge_capacities[edge]);
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

        for (const auto& edge : OutgoingEdges(graph, next_vertex)) {
            visitor.ExamineEdge(edge);
            Vertex target = GetTarget(edge);

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

    for (const auto& edge : OutgoingEdges(graph, origin_vertex)) {
        visitor.ExamineEdge(edge);
        Vertex target = GetTarget(edge);

        if (used.find(target) == used.end()) {
            used.insert(target);
            DFS(target, graph, visitor, used);

            visitor.FinishEdge(edge);
        }
    }
}

int FindMaxFlow(std::shared_ptr<FlowNetwork> net) {
    int max_flow = 0;

    LevelBFShVisitor level_visitor(net->GetVertexCount(), net->GetSource(),
                                   net->GetTarget());

    PushedDFSVisitor pushed_visitor(net);

    auto level_predicate = [&net, &level_visitor](const Edge& edge) -> bool {
        EdgeProperties& edge_properties = net->GetEdgeProperties(edge);

        return !level_visitor.IsFoundVertex(edge.to) &&
               edge_properties.ResidualCap() > 0;
    };

    auto push_predicate = [&level_visitor, &pushed_visitor,
                           &net](const Edge& edge) -> bool {
        EdgeProperties& edge_properties = net->GetEdgeProperties(edge);

        return !pushed_visitor.ReachedTarget() &&
               edge_properties.ResidualCap() > 0 &&
               level_visitor.GetLevel(edge.to) ==
                   level_visitor.GetLevel(edge.from) + 1;
    };

    auto graph = net->ResidualNetworkView(level_predicate);
    auto level_graph = net->ResidualNetworkView(push_predicate);

    do {
        BFS(net->GetSource(), graph, level_visitor);
        int pushed = 0;

        do {
            DFS(net->GetSource(), level_graph, pushed_visitor);
            pushed = pushed_visitor.PushedFlow();
            max_flow += pushed;
        } while (pushed > 0);

    } while (level_visitor.HasPathToTarget());

    return max_flow;
}

template <typename Predicate>
int BinarySearch(Predicate predicate, int left, int right) {
    while (left + 1 != right) {
        int middle = left + (right - left) / 2;
        if (predicate(middle)) {
            right = middle;
        } else {
            left = middle;
        }
    }

    return right;
}

void SetMaxTargetCapacity(std::shared_ptr<FlowNetwork> net,
                          int target_capacity) {
    for (int node = net->GetSource() + 1; node != net->GetTarget(); ++node) {
        net->SetEdgeCapacity({node, net->GetTarget()}, target_capacity);
    }
}

std::shared_ptr<FlowNetwork> BuildFlowNetwork(
    const std::vector<bool>& is_first_part, const std::vector<Name>& names) {
    BuilderGoldIngotFlowNetwork builder;

    int count_vertex = is_first_part.size() + 1;
    int source = 0;
    int target = count_vertex - 1;

    std::vector<Edge> edges;
    std::vector<int64_t> edge_capacities;

    for (int node = 1; node != target; ++node) {
        if (is_first_part[node]) {
            edges.emplace_back(source, node);
            edge_capacities.push_back(1);
        } else {
            edges.emplace_back(node, target);
            edge_capacities.push_back(1);
        }
    }

    for (const auto& [first_word, second_word] : names) {
        edges.emplace_back(first_word, second_word);
        edge_capacities.push_back(count_vertex);
    }

    builder.BuildFlowNetwork(source, target, count_vertex, edges,
                             edge_capacities);

    return builder.GetFlowNetwork();
}

int FindMaxCountChiters(const std::vector<bool>& is_first_part,
                        const std::vector<Name>& names) {
    auto net = BuildFlowNetwork(is_first_part, names);
    int max_flow = FindMaxFlow(net);
    int unique_words = net->GetUniqueCount(is_first_part);

    return names.size() - max_flow - unique_words;
}

struct InputData {
    std::vector<bool> is_first_part;
    std::vector<Name> names;
};

InputData Input(std::istream& input = std::cin) {
    int count_names;
    input >> count_names;

    InputData input_data;

    std::unordered_map<size_t, int> first_part_word_number;
    std::unordered_map<size_t, int> second_part_word_number;

    int word_number = 1;
    input_data.is_first_part.push_back(true);

    auto hasher = std::hash<std::string>{};

    for (int i = 1; i <= count_names; ++i) {
        std::string first_word;
        std::string second_word;
        input >> first_word >> second_word;

        size_t first_word_hash = hasher(first_word);
        size_t second_word_hash = hasher(second_word);

        int first;
        int second;

        if (auto fir = first_part_word_number.find(first_word_hash);
            fir != first_part_word_number.end()) {
            first = fir->second;
        } else {
            first = word_number++;
            input_data.is_first_part.push_back(true);
            first_part_word_number[first_word_hash] = first;
        }

        if (auto sec = second_part_word_number.find(second_word_hash);
            sec != second_part_word_number.end()) {
            second = sec->second;
        } else {
            second = word_number++;
            input_data.is_first_part.push_back(false);
            second_part_word_number[second_word_hash] = second;
        }

        input_data.names.emplace_back(first, second);
    }

    return input_data;
}

void Output(int max_count_chiters, int test, std::ostream& output = std::cout) {
    output << "Case #" << test << ": " << max_count_chiters << std::endl;
}

int main() {
    int count_tests;
    std::cin >> count_tests;
    for (int test = 1; test <= count_tests; ++test) {
        InputData input = Input();

        std::vector<bool> is_first_part = std::move(input.is_first_part);
        std::vector<Name> names = std::move(input.names);

        Output(FindMaxCountChiters(is_first_part, names), test);
    }

    return 0;
}
