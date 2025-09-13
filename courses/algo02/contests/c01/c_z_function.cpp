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

std::string Input() {
    std::string str;
    std::cin >> str;
    return str;
}

template <typename T>
void Output(const std::vector<T>& vec) {
    for (size_t lcp : vec) {
        std::cout << lcp << ' ';
    }
    std::cout << '\n';
}

int main() {
    std::string str = Input();

    std::vector<size_t> vec = ArrayZFunction(str);

    Output(vec);

    return 0;
}
