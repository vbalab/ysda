#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>

std::vector<int> BuildSuffixArray(const std::string& input_str) {
    const int kLength = static_cast<int>(input_str.size());

    std::vector<int> suffix_array(kLength);
    std::vector<int> rank_array(kLength);
    std::vector<int> temp_rank(kLength);

    for (int index = 0; index < kLength; ++index) {
        suffix_array[index] = index;
        rank_array[index] = input_str[index];
    }

    for (int step = 1; step < kLength; step *= 2) {
        auto cmp = [&](int index1, int index2) -> bool {
            if (rank_array[index1] != rank_array[index2]) {
                return rank_array[index1] < rank_array[index2];
            }

            const int kRi =
                (index1 + step < kLength ? rank_array[index1 + step] : -1);
            const int kRj =
                (index2 + step < kLength ? rank_array[index2 + step] : -1);

            return kRi < kRj;
        };

        std::sort(suffix_array.begin(), suffix_array.end(), cmp);

        temp_rank[suffix_array[0]] = 0;

        for (int index = 1; index < kLength; ++index) {
            temp_rank[suffix_array[index]] =
                temp_rank[suffix_array[index - 1]] +
                (cmp(suffix_array[index - 1], suffix_array[index]) ? 1 : 0);
        }

        rank_array = temp_rank;

        if (rank_array[suffix_array[kLength - 1]] == kLength - 1) {
            break;
        }
    }

    return suffix_array;
}

std::vector<int> BuildLCP(const std::string& input_str,
                          const std::vector<int>& suffix_array) {
    const int kLength = static_cast<int>(input_str.size());

    std::vector<int> lcp_array(kLength - 1, 0);
    std::vector<int> pos_array(kLength, 0);

    for (int index = 0; index < kLength; ++index) {
        pos_array[suffix_array[index]] = index;
    }

    int common_length = 0;

    for (int index = 0; index < kLength; ++index) {
        if (pos_array[index] == kLength - 1) {
            common_length = 0;
            continue;
        }

        const int kIndexJ = suffix_array[pos_array[index] + 1];

        while ((index + common_length < kLength) &&
               (kIndexJ + common_length < kLength) &&
               (input_str[index + common_length] ==
                input_str[kIndexJ + common_length])) {
            ++common_length;
        }

        lcp_array[pos_array[index]] = common_length;

        if (common_length > 0) {
            --common_length;
        }
    }

    return lcp_array;
}

class SparseTable {
public:
    explicit SparseTable(const std::vector<int>& arr) {
        const int kLength = static_cast<int>(arr.size());

        log_.resize(kLength + 1);
        log_[1] = 0;

        for (int index = 2; index <= kLength; ++index) {
            log_[index] = log_[index / 2] + 1;
        }

        const int kMax = log_[kLength] + 1;
        st_.clear();
        st_.resize(kLength, std::vector<int>(kMax));

        for (int index = 0; index < kLength; ++index) {
            st_[index][0] = arr[index];
        }

        for (int index_j = 1; index_j < kMax; ++index_j) {
            for (int index = 0; index + (1 << index_j) <= kLength; ++index) {
                st_[index][index_j] =
                    std::min(st_[index][index_j - 1],
                             st_[index + (1 << (index_j - 1))][index_j - 1]);
            }
        }
    }

    int Query(int l_bound, int r_bound) const {
        assert(l_bound <= r_bound);

        const int kIndexJ = log_[r_bound - l_bound + 1];

        return std::min(st_[l_bound][kIndexJ],
                        st_[r_bound - (1 << kIndexJ) + 1][kIndexJ]);
    }

private:
    std::vector<std::vector<int>> st_;
    std::vector<int> log_;
};

// Helper function to compute LPF value for a given index.
int GetLpfForIndex(int index, const std::vector<int>& rank_array,
                   const SparseTable* sparse_table, std::set<int>& bst) {
    const int kPosition = rank_array[index];
    int best = 0;
    if (!bst.empty()) {
        auto it = bst.lower_bound(kPosition);
        if (it != bst.end()) {
            const int kCandidateRank = *it;
            const int kLeftIndex = std::min(kPosition, kCandidateRank);
            const int kRightIndex = std::max(kPosition, kCandidateRank) - 1;
            if ((kLeftIndex <= kRightIndex) && (sparse_table != nullptr)) {
                best = std::max(best,
                                sparse_table->Query(kLeftIndex, kRightIndex));
            }
        }
        if (it != bst.begin()) {
            auto it_prev = std::prev(it);
            const int kCandidateRank = *it_prev;
            const int kLeftIndex = std::min(kPosition, kCandidateRank);
            const int kRightIndex = std::max(kPosition, kCandidateRank) - 1;
            if ((kLeftIndex <= kRightIndex) && (sparse_table != nullptr)) {
                best = std::max(best,
                                sparse_table->Query(kLeftIndex, kRightIndex));
            }
        }
    }
    return best;
}

std::vector<int> ComputeLPF(const std::string& input_str) {
    const int kLength = static_cast<int>(input_str.size());
    const std::vector<int> kSuffixArray = BuildSuffixArray(input_str);

    std::vector<int> rank_array(kLength);

    for (int index = 0; index < kLength; ++index) {
        rank_array[kSuffixArray[index]] = index;
    }

    std::vector<int> lcp_array;

    if (kLength > 1) {
        lcp_array = BuildLCP(input_str, kSuffixArray);
    }

    SparseTable* sparse_table = nullptr;

    if (!lcp_array.empty()) {
        sparse_table = new SparseTable(lcp_array);
    }

    std::vector<int> lpf_array(kLength, 0);
    std::set<int> bst;

    for (int index = 0; index < kLength; ++index) {
        const int kBest = GetLpfForIndex(index, rank_array, sparse_table, bst);
        lpf_array[index] = kBest;
        bst.insert(rank_array[index]);
    }

    delete sparse_table;

    return lpf_array;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string input_str;
    std::cin >> input_str;

    const std::vector<int> kLpfArray = ComputeLPF(input_str);

    for (const int kCurrentValue : kLpfArray) {
        std::cout << kCurrentValue << "\n";
    }

    return 0;
}
