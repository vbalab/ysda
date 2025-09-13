#include <iostream>
#include <string>
#include <vector>

std::vector<size_t> ArrayZFunction(const std::string& str) {
    size_t size = str.size();

    std::vector<size_t> zfun(size, 0);
    zfun[0] = size;

    size_t left = 0;
    size_t right = 0;

    for (size_t i = 1; i < size; ++i) {
        if (i < right) {
            zfun[i] = std::min(zfun[i - left], right - i);
        }

        while (i + zfun[i] < size && str[zfun[i]] == str[i + zfun[i]]) {
            ++zfun[i];
        }

        if (i + zfun[i] > right) {
            right = i + zfun[i];
            left = i;
        }
    }

    return zfun;
}

size_t Solution(const std::string& str) {
    size_t k_max = 1;
    size_t size = str.size();

    for (size_t i = 0; i < size; ++i) {
        std::vector<size_t> zfun = ArrayZFunction(str.substr(i, size - i));

        for (size_t j = 1; j < (size - i) / 2; ++j) {
            if (zfun[j] / j + 1 > k_max) {
                k_max = zfun[j] / j + 1;
            }
        }
    }

    return k_max;
}

std::string Input() {
    std::string str;
    std::cin >> str;
    return str;
}

int main() {
    std::string str = Input();

    std::cout << Solution(str) << '\n';

    return 0;
}
