#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>

constexpr double kAccuracy = 0.00001;

struct Edge {
    int32_t from;
    int32_t to;

    bool operator==(const Edge& other) const = default;
};

struct EdgeProperties {
    int16_t from;
    int16_t to;

    double flow;
    double capacity;
};

class Graph {
public:
    Graph(int16_t count_vertex)
        : count_vertex_(count_vertex), graph_(count_vertex) {}

    void AddEdge(int16_t from, int16_t to, double flow, double capacity) {
        graph_[from].emplace_back(from, to, flow, capacity);
    }

    std::vector<EdgeProperties>& OutgoingEdges(int16_t vertex) {
        return graph_[vertex];
    }

    const std::vector<EdgeProperties>& OutgoingEdges(int16_t vertex) const {
        return graph_[vertex];
    }

    void Sort() {
        for (auto& edges : graph_) {
            std::sort(
                edges.begin(), edges.end(),
                [](const EdgeProperties& first, const EdgeProperties& second) {
                    return first.to < second.to;
                });
        }
    }

    void SetAllFlowZero() {
        for (auto& edges : graph_) {
            for (auto& props : edges) {
                props.flow = 0;
            }
        }
    }

    int16_t GetCountVertex() const { return count_vertex_; }

    EdgeProperties& GetEdgeProperties(int16_t from, int16_t to) {
        for (auto& props : graph_[from]) {
            if (props.to == to) {
                return props;
            }
        }
        throw std::out_of_range("no such vertex");
    }

private:
    int16_t count_vertex_;
    std::vector<std::vector<EdgeProperties>> graph_;
};

template <typename Predicate>
class FilteredIterator {
public:
    class Iterator {
    public:
        Iterator(std::vector<EdgeProperties>& edges, Predicate predicate,
                 size_t index)
            : edges_(edges), predicate_(predicate), current_index_(index) {}

        void Next() {
            while (current_index_ != edges_.size() &&
                   !predicate_(edges_[current_index_])) {
                ++current_index_;
            }
        }

        Iterator operator++() {
            current_index_++;
            Next();
            return Iterator(edges_, predicate_, current_index_);
        }

        bool operator!=(const Iterator& other) const {
            return other.current_index_ != current_index_;
        }

        EdgeProperties& operator*() const { return edges_[current_index_]; }

    private:
        std::vector<EdgeProperties>& edges_;
        Predicate predicate_;
        size_t current_index_;
    };

    Iterator begin() const {  // NOLINT
        auto iterator = Iterator(edges_, predicate_, 0);
        iterator.Next();

        return iterator;
    }
    Iterator end() const {  // NOLINT
        return Iterator(edges_, predicate_, edges_.size());
    }

    FilteredIterator(Predicate predicate, std::vector<EdgeProperties>& edges)
        : predicate_(predicate), edges_(edges) {}

private:
    Predicate predicate_;
    std::vector<EdgeProperties>& edges_;
};

template <typename Predicate>
class FilteredGraph {
public:
    FilteredGraph(std::shared_ptr<Graph> graph, Predicate predicate)
        : graph_(graph), predicate_(predicate) {}

    void AddEdge(int16_t from, int16_t to, double flow, double capacity) {
        graph_->AddEdge(from, to, flow, capacity);
    }

    int16_t GetCountVertex() const { return graph_->GetCountVertex(); }

    FilteredIterator<Predicate> OutgoingEdges(int16_t vertex) {
        return FilteredIterator<Predicate>(predicate_,
                                           graph_->OutgoingEdges(vertex));
    }

private:
    std::shared_ptr<Graph> graph_;
    Predicate predicate_;
};

std::vector<EdgeProperties>& OutgoingEdges(Graph& graph, int16_t vertex) {
    return graph.OutgoingEdges(vertex);
}

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
    virtual void FinishEdge(Edge& /*edge*/) {}
    virtual ~DFSVisitor() = default;
};

class LevelBFShVisitor : public BFSVisitor<int16_t, EdgeProperties> {
public:
    LevelBFShVisitor(int16_t count_vertex, int16_t source, int16_t target)
        : source_(source),
          target_(target),
          count_vertex_(count_vertex),
          level_(count_vertex) {}

    void DiscoverVertex(int16_t vertex) override {
        if (vertex == source_) {
            level_.assign(count_vertex_, -1);
            level_[source_] = 0;
        }
    }

    void ExamineEdge(const EdgeProperties& props) override {
        level_[props.to] = level_[props.from] + 1;
    }

    bool HasPathToTarget() const { return level_[target_] != -1; }

    int16_t GetLevel(int16_t vertex) const { return level_[vertex]; }

private:
    int16_t source_;
    int16_t target_;
    int16_t count_vertex_;
    std::vector<int16_t> level_;
};

class IBuilderFlowNetwork;

class FlowNetwork {
    friend IBuilderFlowNetwork;

public:
    FlowNetwork() = default;

    void SetMaxSourceCapacity(double source_capacity) {
        for (auto& props : graph->OutgoingEdges(source)) {
            props.capacity = source_capacity;
        }
    }

    void SetAllFlowZero() { graph->SetAllFlowZero(); }

    template <typename Predicate>
    FilteredGraph<Predicate> ResidualNetworkView(Predicate predicate) const {
        return FilteredGraph<Predicate>(graph, predicate);
    }

    int16_t source;
    int16_t target;
    std::shared_ptr<Graph> graph;
};

void PrintGraph(std::shared_ptr<FlowNetwork> net) {
    for (int from = net->source; from <= net->target; ++from) {
        for (auto& props : OutgoingEdges(*net->graph, from)) {
            std::cout << props.from << " -(" << props.flow << "/"
                      << props.capacity << ")-> " << props.to << std::endl;
        }
    }
}

class IBuilderFlowNetwork {
public:
    std::shared_ptr<FlowNetwork> GetFlowNetwork() { return net_; }

protected:
    std::shared_ptr<FlowNetwork> net_;
};

class PushedDFSVisitor : public DFSVisitor<int16_t, EdgeProperties> {
public:
    PushedDFSVisitor(int16_t count_vertex, std::shared_ptr<FlowNetwork> net)
        : count_vertex_(count_vertex), pushed_(count_vertex), net_(net) {}

    void DiscoverVertex(int16_t vertex) override {
        if (vertex == net_->source) {
            pushed_.assign(count_vertex_, 0);
        }
    }

    void ExamineEdge(const EdgeProperties& props) override {
        if (props.from == net_->source) {
            pushed_[props.to] = props.capacity - props.flow;
            return;
        }

        pushed_[props.to] =
            std::min(pushed_[props.from], props.capacity - props.flow);
    }

    void FinishEdge(EdgeProperties& props) override {
        if (pushed_[net_->target] == 0) {
            return;
        }

        pushed_[props.from] =
            std::min({props.capacity - props.flow, pushed_[props.to]});

        props.flow += pushed_[props.from];
        net_->graph->GetEdgeProperties(props.to, props.from).flow -=
            pushed_[props.from];
    }

    double PushedFlow() const { return pushed_[net_->source]; }

    bool ReachedTarget() const { return pushed_[net_->target] != 0; }

private:
    int16_t count_vertex_;
    std::vector<double> pushed_;
    std::shared_ptr<FlowNetwork> net_;
};

template <typename Predicate>
FilteredIterator<Predicate> OutgoingEdges(FilteredGraph<Predicate>& graph,
                                          int16_t vertex) {
    return graph.OutgoingEdges(vertex);
}

int16_t GetTarget(const EdgeProperties& props) { return props.to; }

class BuilderGoldIngotFlowNetwork : public IBuilderFlowNetwork {
public:
    void BuildFlowNetwork(int16_t count_people,
                          const std::vector<Edge>& relationship) {
        int16_t count_vertex = count_people + relationship.size() + 2;

        auto new_net = std::make_shared<FlowNetwork>();

        new_net->graph = std::make_shared<Graph>(count_vertex);
        int16_t target = count_vertex - 1;

        const int16_t kSize = static_cast<int16_t>(relationship.size());
        for (int16_t i = 1; i <= kSize; ++i) {
            int16_t edge_node = count_people + i;

            AddEdge(edge_node, target, 1, new_net);

            AddEdge(relationship[i - 1].from, edge_node, count_people, new_net);

            AddEdge(relationship[i - 1].to, edge_node, count_people, new_net);
        }

        for (int16_t i = 1; i <= count_people; ++i) {
            AddEdge(0, i, 0, new_net);
        }

        new_net->source = 0;
        new_net->target = count_vertex - 1;

        net_ = new_net;
    }
    void static AddEdge(int16_t from, int16_t to, double capacity,
                        std::shared_ptr<FlowNetwork> net) {
        net->graph->AddEdge(from, to, 0, capacity);
        net->graph->AddEdge(to, from, 0, 0);
    }
};

template <typename Visitor, typename Vertex, typename Graph>
void BFS(Vertex origin_vertex, Graph& graph, Visitor& visitor) {
    std::vector<bool> used(graph.GetCountVertex(), false);
    std::deque<Vertex> queue;

    visitor.DiscoverVertex(origin_vertex);
    queue.push_back(origin_vertex);
    used[origin_vertex] = true;

    while (!queue.empty()) {
        Vertex next = queue.front();
        queue.pop_front();

        visitor.ExamineVertex(next);

        for (auto& edge : OutgoingEdges(graph, next)) {
            visitor.ExamineEdge(edge);
            Vertex target = GetTarget(edge);

            if (!used[target]) {
                visitor.DiscoverVertex(target);
                used[target] = true;
                queue.push_back(target);
            }
        }
    }
}

template <typename Visitor, typename Vertex, typename Graph>
void DFS(Vertex origin_vertex, Graph& graph, Visitor& visitor) {
    std::vector<bool> used(graph.GetCountVertex(), false);
    DFS(origin_vertex, graph, visitor, used);
}

template <typename Visitor, typename Vertex, typename Graph>
void DFS(Vertex origin_vertex, Graph& graph, Visitor& visitor,
         std::vector<bool>& used) {
    visitor.DiscoverVertex(origin_vertex);
    used[origin_vertex] = true;

    for (auto& edge : OutgoingEdges(graph, origin_vertex)) {
        visitor.ExamineEdge(edge);
        Vertex target = GetTarget(edge);

        if (!used[target]) {
            DFS(target, graph, visitor, used);
            visitor.FinishEdge(edge);
        }
    }
}

double FindMaxFlow(std::shared_ptr<FlowNetwork> net) {
    double max_flow = 0;
    int16_t count_vertex = net->graph->GetCountVertex();

    LevelBFShVisitor level_visitor(count_vertex, net->source, net->target);

    PushedDFSVisitor pushed_visitor(count_vertex, net);

    auto level_predicate =
        [&level_visitor](const EdgeProperties& props) -> bool {
        return level_visitor.GetLevel(props.to) == -1 &&
               props.flow < props.capacity;
    };

    auto push_predicate = [&level_visitor,
                           &pushed_visitor](EdgeProperties& props) -> bool {
        return !pushed_visitor.ReachedTarget() && props.flow < props.capacity &&
               level_visitor.GetLevel(props.to) ==
                   level_visitor.GetLevel(props.from) + 1;
    };

    auto graph = net->ResidualNetworkView(level_predicate);
    auto level_graph = net->ResidualNetworkView(push_predicate);

    do {
        BFS(net->source, graph, level_visitor);
        double pushed = 0;

        do {
            DFS(net->source, level_graph, pushed_visitor);
            pushed = pushed_visitor.PushedFlow();
            max_flow += pushed;
        } while (pushed > 0);
    } while (level_visitor.HasPathToTarget());

    return max_flow;
}

template <typename Predicate>
double BinarySearch(Predicate condition, double left, double right) {
    constexpr int16_t kSize = 100;

    for (int16_t i = 0; i < kSize; ++i) {
        double middle = (left + right) / 2;
        if (condition(middle)) {
            left = middle;
        } else {
            right = middle;
        }
    }

    return right;
}

double FindMaxToxicity(int16_t count_edge, std::shared_ptr<FlowNetwork> net) {
    auto predicate = [&net, count_edge](double toxicity) -> bool {
        // net->SetMaxTargetCapacity(toxicity);
        net->SetMaxSourceCapacity(toxicity);
        net->SetAllFlowZero();

        return FindMaxFlow(net) < count_edge;
    };

    return BinarySearch(predicate, 0, count_edge);
}

class FindUselessVisitor : public DFSVisitor<int16_t, EdgeProperties> {
public:
    FindUselessVisitor(std::shared_ptr<FlowNetwork> net, int16_t count_people)
        : useless_vertex(count_people + 1, false), net_(net) {}

    void DiscoverVertex(int16_t vertex) override {
        if (0 < vertex &&
            vertex < static_cast<int16_t>(useless_vertex.size())) {
            useless_vertex[vertex] = true;
        }
    }

    std::vector<bool> useless_vertex;

private:
    std::shared_ptr<FlowNetwork> net_;
};

std::vector<int16_t> GetMaxToxicityComand(int16_t count_people,
                                          std::vector<Edge>& relationship) {
    if (relationship.empty()) {
        return {1};
    }

    BuilderGoldIngotFlowNetwork builder;
    builder.BuildFlowNetwork(count_people, relationship);

    auto net = builder.GetFlowNetwork();

    double toxicity = FindMaxToxicity(relationship.size(), net);

    net->SetMaxSourceCapacity(toxicity);
    net->SetAllFlowZero();
    FindMaxFlow(net);

    auto predicate = [](const EdgeProperties& props) {
        return std::abs(props.capacity - props.flow) > kAccuracy;
    };

    FindUselessVisitor visitor(net, count_people);
    auto filter_graph = net->ResidualNetworkView(predicate);
    DFS(net->source, filter_graph, visitor);

    std::vector<int16_t> max_toxicity_comand;

    for (int16_t i = 1; i <= count_people; ++i) {
        if (!visitor.useless_vertex[i]) {
            max_toxicity_comand.push_back(i);
        }
    }

    return max_toxicity_comand;
}

struct InputData {
    InputData(int16_t count_people) : count_people(count_people) {}
    int16_t count_people;
    std::vector<Edge> relationship;
};

InputData Input(std::istream& input = std::cin) {
    int16_t count_people;
    int16_t count_pair;
    input >> count_people >> count_pair;

    InputData input_data(count_people);

    for (int16_t i = 0; i < count_pair; ++i) {
        int16_t from;
        int16_t to;
        input >> from >> to;
        input_data.relationship.emplace_back(from, to);
    }

    return input_data;
}

void Output(std::vector<int16_t> answer) {
    std::cout << answer.size() << std::endl;
    std::sort(answer.begin(), answer.end());

    for (auto node : answer) {
        std::cout << node << std::endl;
    }
}

int main() {
    auto input_data = Input(std::cin);

    int16_t count_people = std::move(input_data.count_people);
    std::vector<Edge> relationship = std::move(input_data.relationship);

    Output(GetMaxToxicityComand(count_people, relationship));

    return 0;
}
