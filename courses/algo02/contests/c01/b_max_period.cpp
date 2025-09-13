#include <iostream>
#include <string>
#include <vector>

std::vector<size_t> ArrayLPS(const std::string& str) {
    std::vector<size_t> prefixes(str.size(), 0);

    for (size_t i = 1; i < str.size(); ++i) {
        size_t last = prefixes[i - 1];

        while (last > 0 && str[last] != str[i]) {
            last = prefixes[last - 1];
        }

        prefixes[i] = last + static_cast<size_t>(str[last] == str[i]);
    }

    return prefixes;
}

size_t FindMaxPeriod(const std::string& str) {
    std::vector<size_t> lps = ArrayLPS(str);
    size_t len = str.size() - lps.back();

    if (len != 0 && str.size() % len == 0) {
        return str.size() / len;
    }

    return 1;
}

std::string Input() {
    std::string str;
    std::cin >> str;
    return str;
}

int main() {
    std::string str = Input();

    std::cout << FindMaxPeriod(str) << '\n';

    return 0;
}
