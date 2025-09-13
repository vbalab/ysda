#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

std::vector<int> BuildSuffixArray(const std::string& str) {
    int size = str.size();
    std::vector<int> sa(size);
    std::vector<int> rank(size);

    for (int i = 0; i < size; i++) {
        sa[i] = i;
        rank[i] = str[i];
    }

    std::vector<int> temp_rank(size);

    for (int step = 1; step < size; step *= 2) {
        auto cmp = [&](int left, int right) -> bool {
            if (rank[left] != rank[right]) {
                return rank[left] < rank[right];
            }

            return rank[(left + step) % size] < rank[(right + step) % size];
        };

        std::sort(sa.begin(), sa.end(), cmp);

        temp_rank[sa[0]] = 0;
        for (int i = 1; i < size; i++) {
            temp_rank[sa[i]] =
                temp_rank[sa[i - 1]] + (cmp(sa[i - 1], sa[i]) ? 1 : 0);
        }
        rank = temp_rank;

        if (rank[sa[size - 1]] == size - 1) {
            break;
        }
    }
    return sa;
}

std::string ComputeBwt(const std::string& str) {
    int size = str.size();
    std::vector<int> sa = BuildSuffixArray(str);
    std::string bwt;

    bwt.resize(size);

    for (int i = 0; i < size; i++) {
        int pos = sa[i];
        bwt[i] = str[(pos + size - 1) % size];
    }

    return bwt;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string str;
    std::cin >> str;

    std::cout << ComputeBwt(str) << "\n";
    return 0;
}
