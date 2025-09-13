#include <bits/stdc++.h>

struct Point {
    long long x;
    long long y;
};

constexpr int kPresicion = 10;

void ReadAgents(int& number_of_agents, std::vector<Point>& agents) {
    std::cin >> number_of_agents;
    agents.resize(number_of_agents);
    for (int i = 0; i < number_of_agents; i++) {
        std::cin >> agents[i].x >> agents[i].y;
    }
}

double ComputeMinimalRadius(int number_of_agents,
                            const std::vector<Point>& agents) {
    if (number_of_agents == 1) {
        return 0.0;
    }

    std::vector<double> dist(number_of_agents,
                             std::numeric_limits<double>::infinity());
    std::vector<bool> in_mst(number_of_agents, false);
    dist[0] = 0.0;
    double max_edge = 0.0;

    for (int i = 0; i < number_of_agents; i++) {
        int current_node = -1;
        double best = std::numeric_limits<double>::infinity();
        for (int j = 0; j < number_of_agents; j++) {
            if (!in_mst[j] && dist[j] < best) {
                best = dist[j];
                current_node = j;
            }
        }

        in_mst[current_node] = true;
        if (current_node != 0) {
            max_edge = std::max(max_edge, dist[current_node]);
        }

        for (int neighbor_node = 0; neighbor_node < number_of_agents;
             neighbor_node++) {
            if (!in_mst[neighbor_node]) {
                long long dx = agents[current_node].x - agents[neighbor_node].x;
                long long dy = agents[current_node].y - agents[neighbor_node].y;
                double distance_between_nodes =
                    std::sqrt((long double)dx * dx + (long double)dy * dy);
                if (distance_between_nodes < dist[neighbor_node]) {
                    dist[neighbor_node] = distance_between_nodes;
                }
            }
        }
    }

    return max_edge;
}

void PrintResult(double result) {
    std::cout << std::fixed << std::setprecision(kPresicion) << result << "\n";
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int number_of_agents;
    std::vector<Point> agents;
    ReadAgents(number_of_agents, agents);

    double minimal_radius = ComputeMinimalRadius(number_of_agents, agents);
    PrintResult(minimal_radius);

    return 0;
}
