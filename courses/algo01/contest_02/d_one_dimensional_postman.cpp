#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

template <typename UT>
class RandomGenerator {
private:
    UT a_;
    UT b_;
    UT cur_;
    static const UT kShiftAmount = 8;

public:
    RandomGenerator(UT par_a, UT par_b) : a_(par_a), b_(par_b), cur_(0) {}

    UT NextRand24() {
        this->cur_ = this->cur_ * this->a_ + this->b_;
        return this->cur_ >> kShiftAmount;
    }

    UT NextRand32() {
        UT par_a = NextRand24();
        UT par_b = NextRand24();
        return (par_a << kShiftAmount) ^ par_b;
    }
};

template <typename T>
uint64_t LossFunction(const std::vector<T>& homes, T point) {
    uint64_t loss = 0;
    for (const T& home : homes) {
        if (home > point) {
            loss += static_cast<uint64_t>(home - point);
        } else {
            loss += static_cast<uint64_t>(point - home);
        }
    }

    return loss;
}

// uint64_t ModifiedBinarySearch(const std::vector<uint32_t>& homes, uint32_t
// left,
//                               uint32_t right) {
//     if (left >= right) {
//         return left;
//     }

//     uint32_t middle = (left + right) / 2;
//     uint64_t exact = LossFunction(homes, middle);

//     uint64_t before = LossFunction(homes, middle);
//     if (exact > before) {
//         return ModifiedBinarySearch(homes, left, middle - 1);
//     }

//     uint64_t after = LossFunction(homes, middle + 1);
//     if (exact > after) {
//         return ModifiedBinarySearch(homes, middle + 1, right);
//     }

//     return exact;
// }

template <typename Iterator>
void InsertionSort(Iterator begin, Iterator end) {
    for (Iterator i = begin + 1; i != end; ++i) {
        auto key = *i;
        Iterator lower = i;

        while (lower != begin && *(lower - 1) > key) {
            *lower = *(lower - 1);
            --lower;
        }
        *lower = key;
    }
}

// In-place
template <typename Iterator>
typename std::iterator_traits<Iterator>::value_type MedianOfFives(
    Iterator begin, Iterator end) {
    std::size_t size = std::distance(begin, end);
    if (size <= 5) {
        InsertionSort(begin, end);
        return *(begin + size / 2);
    }

    Iterator medians_iter = begin;
    for (Iterator group_start = begin; group_start < end; group_start += 5) {
        Iterator group_end = group_start + 5;
        if (group_end > end) {
            group_end = end;
        }

        InsertionSort(group_start, group_end);
        Iterator median = group_start + (group_end - group_start) / 2;
        std::iter_swap(medians_iter, median);
        ++medians_iter;
    }

    return MedianOfFives(begin, medians_iter);
}

template <typename T>
T QuickSelect(std::vector<T>& vec, std::size_t left, std::size_t right,
              std::size_t kth) {
    if (left == right) {
        return vec[left];
    }

    T pivot = MedianOfFives(vec.begin() + left, vec.begin() + right + 1);

    // Lomuto partitioning
    std::size_t split = left;
    for (std::size_t i = left; i <= right; ++i) {
        if (vec[i] < pivot) {
            std::swap(vec[split], vec[i]);
            ++split;
        }
    }

    for (std::size_t i = split; i <= right; ++i) {
        if (vec[i] == pivot) {
            std::swap(vec[split], vec[i]);
            break;
        }
    }

    if (split < kth) {
        return QuickSelect(vec, split + 1, right, kth);
    }
    if (split > kth) {
        return QuickSelect(vec, left, split - 1, kth);
    }
    return pivot;
}

template <typename T>
T QuickSelectMedian(std::vector<T>& vec) {
    std::size_t size = vec.size();
    std::size_t kth = size / 2;

    // if (size % 2 == 0) {
    //     return (QuickSelect(vec, 0, size - 1, kth) +
    //             QuickSelect(vec, 0, size - 1, kth - 1)) /
    //            2;
    // }
    return QuickSelect(vec, 0, size - 1,
                       kth);  // it doesn't matter for the task
}

int main() {
    std::size_t nums;
    uint32_t par_a;
    uint32_t par_b;
    std::cin >> nums >> par_a >> par_b;

    std::vector<uint32_t> homes(nums);
    RandomGenerator<uint32_t> rg(par_a, par_b);

    for (uint32_t& home : homes) {
        home = rg.NextRand32();
    }

    auto start = std::chrono::high_resolution_clock::now();
    uint32_t median = QuickSelectMedian(homes);
    std::cout << LossFunction(homes, median) << '\n';
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Time taken: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms" << std::endl;

    return 0;
}
