#include <algorithm>
#include <climits>
#include <iostream>
#include <stack>
#include <vector>

struct Edge {
    int to;
    int cost;
};

struct Frame {
    int cur_v;
    int parent;
    size_t next_child_idx;
    int edge_cost;
};

class BridgeFinder {
public:
    BridgeFinder(int num_vertices)
        : num_vertices_(num_vertices),
          graph_(num_vertices, std::vector<Edge>()),
          visited_(num_vertices, false),
          tin_(num_vertices, -1),
          low_(num_vertices, -1),
          timer_(0) {}

    void AddEdge(int cur_v, int v_par, int cost) {
        --cur_v;
        --v_par;
        graph_[cur_v].push_back(Edge{v_par, cost});
        graph_[v_par].push_back(Edge{cur_v, cost});
    }

    void FindBridges() {
        for (int cur_v = 0; cur_v < num_vertices_; ++cur_v) {
            if (!visited_[cur_v]) {
                IterativeDFS(cur_v, -1);
            }
        }
    }

    int GetMinimumBridgeCost() const {
        if (bridges_.empty()) {
            return -1;
        }
        return *std::min_element(bridges_.begin(), bridges_.end());
    }

private:
    int num_vertices_;
    std::vector<std::vector<Edge>> graph_;
    std::vector<bool> visited_;
    std::vector<int> tin_;
    std::vector<int> low_;
    int timer_;
    std::vector<int> bridges_;

    void IterativeDFS(int start, int parent) {
        std::stack<Frame> stk;
        stk.push(Frame{start, parent, 0, -1});
        visited_[start] = true;
        tin_[start] = low_[start] = timer_++;

        while (!stk.empty()) {
            Frame& current_frame = stk.top();
            int cur_v = current_frame.cur_v;

            if (current_frame.next_child_idx >= graph_[cur_v].size()) {
                stk.pop();
                if (current_frame.parent != -1 &&
                    current_frame.edge_cost != -1) {
                    low_[current_frame.parent] =
                        std::min(low_[current_frame.parent], low_[cur_v]);
                    if (low_[cur_v] > tin_[current_frame.parent]) {
                        bridges_.push_back(current_frame.edge_cost);
                    }
                }
                continue;
            }

            Edge edge = graph_[cur_v][current_frame.next_child_idx++];
            int v_par = edge.to;
            int edge_cost = edge.cost;

            if (v_par == current_frame.parent) {
                continue;
            }

            if (visited_[v_par]) {
                low_[cur_v] = std::min(low_[cur_v], tin_[v_par]);
            } else {
                stk.push(Frame{v_par, cur_v, 0, edge_cost});
                visited_[v_par] = true;
                tin_[v_par] = low_[v_par] = timer_++;
            }
        }
    }
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int num_vertices;
    int num_edges;
    std::cin >> num_vertices >> num_edges;

    BridgeFinder bridge_finder(num_vertices);

    for (int i = 0; i < num_edges; ++i) {
        int cur_v;
        int v_par;
        int cost;
        std::cin >> cur_v >> v_par >> cost;
        bridge_finder.AddEdge(cur_v, v_par, cost);
    }

    bridge_finder.FindBridges();

    std::cout << bridge_finder.GetMinimumBridgeCost() << '\n';

    return 0;
}
