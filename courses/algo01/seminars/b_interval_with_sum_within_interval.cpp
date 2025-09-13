#include <algorithm>
#include <iostream>
#include <vector>

class MergeSortCounter {
public:
    using Iterator = std::vector<int64_t>::iterator;

    int64_t SortAndCount(Iterator begin, Iterator end, int64_t left_bound,
                         int64_t right_bound) {
        left_side_ = left_bound;
        right_side_ = right_bound;
        result_ = 0;
        MergeSort(begin, end);
        return result_;
    }

private:
    int64_t result_;
    int64_t left_side_;
    int64_t right_side_;

    void MergeSort(Iterator begin, Iterator end) {
        int64_t length = std::distance(begin, end);
        if (length <= 1) {
            return;
        }

        Iterator mid = begin;
        std::advance(mid, length / 2);

        MergeSort(begin, mid);
        MergeSort(mid, end);

        result_ += CountValidSets(begin, mid, end);
        Merge(begin, mid, end);
    }

    void static Merge(Iterator begin, Iterator mid, Iterator end) {
        std::vector<int64_t> temp;
        temp.reserve(std::distance(begin, end));

        Iterator left = begin;
        Iterator right = mid;

        while (left != mid && right != end) {
            if (*left <= *right) {
                temp.push_back(*left++);
            } else {
                temp.push_back(*right++);
            }
        }

        temp.insert(temp.end(), left, mid);
        temp.insert(temp.end(), right, end);

        std::move(temp.begin(), temp.end(), begin);
    }

    int64_t CountValidSets(Iterator begin, Iterator mid, Iterator end) const {
        int64_t count = 0;
        Iterator left = begin;
        Iterator right = mid;
        Iterator right_end = mid;

        while (left != mid) {
            while (right != end && (*right - *left) < left_side_) {
                ++right;
            }
            while (right_end != end && (*right_end - *left) <= right_side_) {
                ++right_end;
            }
            count += std::distance(right, right_end);
            ++left;
        }

        return count;
    }
};

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int64_t num_elements;
    std::cin >> num_elements;

    std::vector<int64_t> array(num_elements);
    for (int64_t& element : array) {
        std::cin >> element;
    }

    int64_t left_bound;
    int64_t right_bound;
    std::cin >> left_bound >> right_bound;

    std::vector<int64_t> prefix_sums(num_elements + 1, 0);
    for (int64_t i = 0; i < num_elements; ++i) {
        prefix_sums[i + 1] = prefix_sums[i] + array[i];
    }

    MergeSortCounter counter;
    int64_t result = counter.SortAndCount(
        prefix_sums.begin(), prefix_sums.end(), left_bound, right_bound);

    std::cout << result << '\n';

    return 0;
}
