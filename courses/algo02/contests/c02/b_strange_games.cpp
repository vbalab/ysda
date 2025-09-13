#include <algorithm>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

void PrintAnswerStatus(int answer_count, int& queries_remaining,
                       int& current_text) {
    std::cout << "$ " << answer_count << std::endl;

    queries_remaining = 5;
    current_text++;
}

std::vector<int> ZFunction(int text_size, int sample_size) {
    std::vector<int> zed(text_size + sample_size + 1, 0);

    int left = 0;
    int right = 1;
    int current_text = 1;
    int answer_count = 0;
    int queries_remaining = 5;

    for (int i = 1; i < text_size + 1; ++i) {
        if (i < right) {
            zed[i] = std::min(zed[i - left], right - i);
        }

        while (zed[i] < sample_size) {
            if (queries_remaining == 0 ||
                i + zed[i] == sample_size + current_text) {
                PrintAnswerStatus(answer_count, queries_remaining,
                                  current_text);
            }

            std::cout << "s " << zed[i] + 1;

            if (zed[i] + i < sample_size) {
                std::cout << " s " << zed[i] + i + 1;
            } else {
                std::cout << " t " << (zed[i] + i - sample_size) + 1;
            }

            std::cout << std::endl;

            std::string ans;
            std::cin >> ans;

            queries_remaining--;

            if (ans == "Yes") {
                zed[i]++;
            } else {
                break;
            }
        }

        if (zed[i] == sample_size && i >= sample_size) {
            answer_count++;
            PrintAnswerStatus(answer_count, queries_remaining, current_text);
        }

        if (zed[i] + i > right) {
            left = i;
            right = left + zed[i];
        }
    }

    while (current_text < text_size + 1) {
        std::cout << "$ " << answer_count << std::endl;
        current_text++;
    }

    return zed;
}

int main() {
    int text_size;
    int sample_size;
    std::cin >> sample_size;
    std::cin >> text_size;

    auto res = ZFunction(text_size, sample_size);

    return 0;
}
