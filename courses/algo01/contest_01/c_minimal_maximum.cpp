#include <algorithm>
#include <iostream>
#include <vector>

std::size_t FindMinMax(const std::vector<int>& A, const std::vector<int>& B, std::size_t left,
                         std::size_t right) {
    if (right - left <= 1) {
        if (std::max(A[left], B[left]) < std::max(A[right], B[right])) {
            return left;
        } else {
            return right;
        }
    }

    std::size_t m = (left + right) / 2;

    if (A[m] >= B[m]) {
        return FindMinMax(A, B, left, m);
    } else {
        return FindMinMax(A, B, m, right);
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::size_t n;
    std::size_t m;
    std::size_t l;

    std::cin >> n >> m >> l;

    std::vector<std::vector<int>> A(n, std::vector<int>(l));
    std::vector<std::vector<int>> B(m, std::vector<int>(l));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < l; ++j) {
            std::cin >> A[i][j];
        }
    }

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < l; ++j) {
            std::cin >> B[i][j];
        }
    }

    std::size_t q;
    std::cin >> q;

    std::vector<std::vector<std::size_t>> Q(q, std::vector<std::size_t>(2));

    for (std::size_t i = 0; i < q; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            std::cin >> Q[i][j];
        }
    }

    for (std::size_t i = 0; i < q; ++i) {
        std::cout << FindMinMax(A[Q[i][0] - 1], B[Q[i][1] - 1], 0, l - 1) + 1 << '\n';
    }

    return 0;
}
