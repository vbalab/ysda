#include <cstdint>
#include <deque>
#include <iostream>
#include <vector>

std::vector<int32_t> MaxInSlidingWindow(const std::vector<int32_t>& values,
                                        const std::vector<char>& turns) {
    std::vector<int32_t> result;

    std::deque<int32_t> maxes;
    maxes.push_back(values[0]);

    std::size_t left = 0;
    std::size_t right = 0;

    for (std::size_t i = 0; i < turns.size(); ++i) {
        if (turns[i] == 'R') {
            ++right;

            if (values[right] <= maxes.front()) {
                while (values[right] > maxes.back()) {
                    maxes.pop_back();
                }

                maxes.push_back(values[right]);
            } else {
                while (!maxes.empty()) {
                    maxes.pop_front();
                }
                maxes.push_back(values[right]);
            }
        } else {
            if (values[left] == maxes.front()) {
                maxes.pop_front();
            }

            ++left;
        }

        result.push_back(maxes.front());
    }

    return result;
}

int main() {
    std::size_t n_values;
    std::size_t m_turns;

    std::cin >> n_values;
    std::vector<int32_t> values(n_values);
    for (int32_t& num : values) {
        std::cin >> num;
    }

    std::cin >> m_turns;
    std::vector<char> turns(m_turns);
    for (char& turn : turns) {
        std::cin >> turn;
    }

    std::vector<int32_t> result = MaxInSlidingWindow(values, turns);

    for (const int32_t& num : result) {
        std::cout << num << ' ';
    }
    std::cout << '\n';

    return 0;
}
