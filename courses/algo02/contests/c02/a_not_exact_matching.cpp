#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

std::vector<size_t> ZFunction(const std::string& str) {
    const size_t kSize = str.size();
    std::vector<size_t> z(kSize);
    size_t left = 0;
    size_t right = 0;
    z[0] = kSize;
    for (size_t i = 1; i < kSize; ++i) {
        if (i < right) {
            z[i] = std::min(right - i, z[i - left]);
        } else {
            z[i] = 0;  // Ensure initialization when not in the window.
        }
        while (i + z[i] < kSize && str[z[i]] == str[i + z[i]]) {
            ++z[i];
        }
        if (i + z[i] > right) {
            left = i;
            right = i + z[i];
        }
    }
    return z;
}

std::vector<size_t> FindMatches(const std::string& pattern,
                                const std::string& text) {
    const size_t kTextSize = text.size();
    const size_t kPatternSize = pattern.size();

    // If the pattern is longer than the text, no matches can be found.
    if (kTextSize < kPatternSize) {
        return std::vector<size_t>();
    }

    const std::string kConcatForward = pattern + "#" + text;
    std::vector<size_t> z_forward = ZFunction(kConcatForward);

    std::string pattern_rev = pattern;
    std::string text_rev = text;
    std::reverse(pattern_rev.begin(), pattern_rev.end());
    std::reverse(text_rev.begin(), text_rev.end());
    const std::string kConcatReverse = pattern_rev + "#" + text_rev;
    std::vector<size_t> z_reverse = ZFunction(kConcatReverse);

    std::vector<size_t> matches;
    for (size_t i = 0; i <= kTextSize - kPatternSize; ++i) {
        if (z_forward[i + kPatternSize + 1] + z_reverse[kTextSize + 1 - i] >=
            kPatternSize - 1) {
            matches.push_back(i + 1);
        }
    }

    return matches;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string pattern;
    std::string text;
    std::cin >> pattern >> text;

    std::vector<size_t> matches = FindMatches(pattern, text);

    std::cout << matches.size() << "\n";
    for (size_t i = 0; i < matches.size(); ++i) {
        std::cout << matches[i] << (i + 1 < matches.size() ? " " : "");
    }

    return 0;
}
