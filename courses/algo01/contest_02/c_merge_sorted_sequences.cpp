#include <cstdint>
#include <iostream>
#include <vector>

std::size_t ExponentialBinarySearch(const std::vector<int32_t>& vec,
                                    int32_t value, std::size_t start,
                                    std::size_t end) {
    if (end - start <= 1) {
        if (value > vec[start]) {
            return start;
        }
        if (value > vec[end]) {
            return end;
        }
        return end + 1;
    }

    std::size_t pow = 2;

    std::size_t left = start;
    std::size_t right = start + pow - 1;
    while (left < end) {
        if (value > vec[left]) {
            return left;
        }
        if (value < vec[left] && value <= vec[right]) {
            return ExponentialBinarySearch(vec, value, left, right);
        }

        pow *= 2;
        left = right;
        right = start + pow - 1;
        if (right >= end) {
            right = end;
        }
    }

    return right + 1;
}

std::vector<int32_t> MergeTwoSortedSequences(const std::vector<int32_t>& vec1,
                                             const std::vector<int32_t>& vec2) {
    std::vector<int32_t> result(vec1.size() + vec2.size());
    std::size_t index_result = 0;

    std::size_t index1 = 0;
    std::size_t index2 = 0;
    std::size_t end1 = vec1.size() - 1;
    std::size_t end2 = vec2.size() - 1;

    std::size_t tmp;

    while (index1 < vec1.size() && index2 < vec2.size()) {
        if (vec1[index1] < vec2[index2]) {
            tmp = ExponentialBinarySearch(vec1, vec2[index2], index1, end1);

            while (index1 <= tmp) {
                result[index_result++] = vec1[index1++];
            }
        } else if (vec1[index1] > vec2[index2]) {
            tmp = ExponentialBinarySearch(vec2, vec1[index1], index2, end2);

            while (index2 <= tmp) {
                result[index_result++] = vec2[index2++];
            }
        } else {
            result[index_result++] = vec1[index1++];
            result[index_result++] = vec2[index2++];
        }
    }

    while (index1 < vec1.size()) {
        result[index_result++] = vec1[index1++];
    }

    while (index2 < vec2.size()) {
        result[index_result++] = vec2[index2++];
    }

    return result;
}

std::vector<int32_t> MergeKSortedSequences(
    const std::vector<std::vector<int32_t>>& sequences, std::size_t left,
    std::size_t right) {
    if (left == right) {
        return sequences[left];
    }

    std::size_t mid = left + (right - left) / 2;
    std::vector<int32_t> left_merged =
        MergeKSortedSequences(sequences, left, mid);
    std::vector<int32_t> right_merged =
        MergeKSortedSequences(sequences, mid + 1, right);

    return MergeTwoSortedSequences(left_merged, right_merged);
}

int main() {
    std::size_t n_vec;
    std::size_t m_vec;
    std::cin >> n_vec >> m_vec;

    std::vector<std::vector<int32_t>> vec(n_vec, std::vector<int32_t>(m_vec));
    for (std::size_t i = 0; i < n_vec; ++i) {
        for (std::size_t j = 0; j < m_vec; ++j) {
            std::cin >> vec[i][j];
        }
    }

    std::vector<int32_t> result = MergeKSortedSequences(vec, 0, n_vec - 1);

    for (int32_t& val : result) {
        std::cout << val << ' ';
    }
    std::cout << '\n';

    return 0;
}
