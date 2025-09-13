#include <algorithm>
#include <climits>
#include <iostream>
#include <queue>
#include <vector>

void BFS(const std::vector<int>& watch_my_flips, int initial_state,
         int target_a, int target_b, int n_by_m) {
    if (initial_state == target_a || initial_state == target_b) {
        std::cout << 0 << '\n';
        return;
    }

    int head = 0;
    int size = 1 << n_by_m;

    std::vector<char> dist(size, -1);
    dist[initial_state] = 0;

    std::vector<int> queue;
    queue.push_back(initial_state);

    while (head < static_cast<int>(queue.size())) {
        int state = queue[head++];
        int distance = dist[state];

        for (int flip : watch_my_flips) {
            int flipped_state = state ^ flip;

            if (dist[flipped_state] == -1) {
                dist[flipped_state] = distance + 1;

                if (flipped_state == target_a || flipped_state == target_b) {
                    std::cout << distance + 1 << '\n';
                    return;
                }

                queue.push_back(flipped_state);
            }
        }
    }

    std::cout << -1 << '\n';
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    int length;
    int height;
    std::cin >> height >> length;

    std::vector<std::vector<char>> plate(height, std::vector<char>(length));
    std::vector<int> watch_my_flips;

    int bitmask_initial = 0;
    int bitmask_a = 0;
    int bitmask_b = 0;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < length; ++j) {
            std::cin >> plate[i][j];

            int index = i * length + j;

            if (plate[i][j] == '1') {
                bitmask_initial |= (1 << index);
            }

            if ((i + j) % 2 == 1) {
                bitmask_a |= (1 << index);
            } else {
                bitmask_b |= (1 << index);
            }

            if (i + 1 < height) {
                watch_my_flips.push_back((1 << index) ^
                                         (1 << ((i + 1) * length + j)));
            }

            if (j + 1 < length) {
                watch_my_flips.push_back((1 << index) ^
                                         (1 << (i * length + j + 1)));
            }
        }
    }

    BFS(watch_my_flips, bitmask_initial, bitmask_a, bitmask_b, length * height);

    return 0;
}
