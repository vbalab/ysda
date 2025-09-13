#include <algorithm>
#include <iostream>
#include <stack>
#include <vector>

class KosarajuSCC {
public:
    KosarajuSCC(int n_vertices, int n_matches)
        : n_vertices_(n_vertices),
          graph_(n_vertices, std::vector<int>()),
          reverse_graph_(n_vertices, std::vector<int>()),
          finish_time_(n_vertices, -1),
          inverse_finish_(n_vertices, -1),
          component_id_(n_vertices, -1) {
        graph_.reserve(n_matches);
        reverse_graph_.reserve(n_matches);
    }

    void InputEdges(int n_matches) {
        // to handle duplicate edges efficiently, sort adjacency lists later
        for (int i = 0; i < n_matches; ++i) {
            int left;
            int right;
            int result;

            std::cin >> left >> right >> result;

            --left;
            --right;

            if (result == 1) {
                graph_[left].push_back(right);
                reverse_graph_[right].push_back(left);
            } else if (result == 2) {
                graph_[right].push_back(left);
                reverse_graph_[left].push_back(right);
            }
        }

        // remove duplicate edges by sorting and using unique
        for (int i = 0; i < n_vertices_; ++i) {
            if (!graph_[i].empty()) {
                std::sort(graph_[i].begin(), graph_[i].end());
                graph_[i].erase(std::unique(graph_[i].begin(), graph_[i].end()),
                                graph_[i].end());
            }
            if (!reverse_graph_[i].empty()) {
                std::sort(reverse_graph_[i].begin(), reverse_graph_[i].end());
                reverse_graph_[i].erase(std::unique(reverse_graph_[i].begin(),
                                                    reverse_graph_[i].end()),
                                        reverse_graph_[i].end());
            }
        }
    }

    void ComputeFinishTimes() {
        int time_counter = 0;
        for (int i = 0; i < n_vertices_; ++i) {
            if (finish_time_[i] == -1) {
                IterativeDFSFirstPass(i, time_counter);
            }
        }
    }

    int Solve() {
        int answer = 0;
        int num_of_components = 0;
        is_root_.resize(n_vertices_, 1);  // initialize all as roots
        component_size_.reserve(n_vertices_);

        for (int i = n_vertices_ - 1; i >= 0; --i) {
            int vertex = inverse_finish_[i];

            if (component_id_[vertex] == -1) {
                component_size_.push_back(0);
                IterativeDFSSecondPass(vertex, num_of_components);

                if (is_root_[num_of_components] != 0) {
                    answer = std::max(
                        answer,
                        n_vertices_ - component_size_[num_of_components] + 1);
                }

                ++num_of_components;
            }
        }

        return answer;
    }

private:
    int n_vertices_;
    std::vector<std::vector<int>> graph_;
    std::vector<std::vector<int>> reverse_graph_;
    std::vector<int> finish_time_;
    std::vector<int> inverse_finish_;
    std::vector<int> component_id_;
    std::vector<char> is_root_;
    std::vector<int> component_size_;

    void IterativeDFSFirstPass(int start, int& time_counter) {
        std::stack<std::pair<int, bool>> stk;
        stk.emplace(start, false);

        while (!stk.empty()) {
            auto [node, processed] = stk.top();
            stk.pop();

            if (node < 0) {  // post-processing
                node = ~node;

                finish_time_[node] = time_counter;
                inverse_finish_[time_counter] = node;

                ++time_counter;
                continue;
            }

            if (finish_time_[node] != -1) {
                continue;
            }

            stk.emplace(~node, false);  // mark for post-processing
            finish_time_[node] = 0;     // temporary mark to indicate visited
            for (auto it = graph_[node].rbegin(); it != graph_[node].rend();
                 ++it) {
                if (finish_time_[*it] == -1) {
                    stk.emplace(*it, false);
                }
            }
        }
    }

    void IterativeDFSSecondPass(int start, int component) {
        std::stack<int> stk;
        stk.push(start);

        component_id_[start] = component;
        component_size_[component] = 1;

        while (!stk.empty()) {
            int node = stk.top();
            stk.pop();

            for (int neighbor : reverse_graph_[node]) {
                if (component_id_[neighbor] == -1) {
                    component_id_[neighbor] = component;
                    ++component_size_[component];

                    stk.push(neighbor);
                } else if (component_id_[neighbor] != component) {
                    is_root_[component] = 0;
                }
            }
        }
    }
};

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n_vertices;
    int n_matches;
    std::cin >> n_vertices >> n_matches;

    KosarajuSCC scc(n_vertices, n_matches);

    scc.InputEdges(n_matches);
    scc.ComputeFinishTimes();

    std::cout << scc.Solve() << "\n";

    return 0;
}
