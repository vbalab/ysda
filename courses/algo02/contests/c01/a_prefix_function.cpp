#include <iostream>
#include <string>
#include <vector>

// Longest Prefix-Suffix
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

std::string Input() {
    std::string str;
    std::cin >> str;
    return str;
}

void Output(const std::vector<size_t>& prefixes) {
    for (size_t i = 0; i < prefixes.size(); ++i) {
        std::cout << prefixes[i] << " ";
    }
    std::cout << '\n';
}

int main() {
    std::string str = Input();

    std::vector<size_t> prefixes = ArrayLPS(str);

    Output(prefixes);

    return 0;
}
