#include <bits/stdc++.h>

struct DSU {
    std::vector<int> parent;
    std::vector<int> size;

    explicit DSU(int vertex_count)
        : parent(vertex_count), size(vertex_count, 1) {
        for (int index = 0; index < vertex_count; ++index) {
            parent[index] = index;
        }
    }

    int FindSet(int vertex_index) {
        if (parent[vertex_index] == vertex_index) {
            return vertex_index;
        }

        parent[vertex_index] = FindSet(parent[vertex_index]);
        return parent[vertex_index];
    }

    bool UnionSet(int first, int second) {
        first = FindSet(first);
        second = FindSet(second);

        if (first == second) {
            return false;
        }

        if (size[first] < size[second]) {
            std::swap(first, second);
        }

        parent[second] = first;
        size[first] += size[second];

        return true;
    }
};

struct Edge {
    int from;
    int to;
    int64_t weight;
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int vertex_count;
    int edge_count;
    std::cin >> vertex_count >> edge_count;

    std::vector<Edge> edges;
    edges.reserve(static_cast<size_t>(edge_count));

    for (int i = 0; i < edge_count; ++i) {
        int start;
        int end;
        int64_t weight;

        std::cin >> start >> end >> weight;
        edges.push_back({start - 1, end - 1, weight});
    }

    std::sort(edges.begin(), edges.end(),
              [](const Edge& first, const Edge& second) {
                  return first.weight < second.weight;
              });

    DSU dsu(vertex_count);

    int64_t answer = 0;
    int added = 0;

    for (auto& ref : edges) {
        if (dsu.UnionSet(ref.from, ref.to)) {
            answer = std::max(answer, ref.weight);
            ++added;

            if (added == vertex_count - 1) {
                break;
            }
        }
    }

    std::cout << answer << "\n";

    return 0;
}
