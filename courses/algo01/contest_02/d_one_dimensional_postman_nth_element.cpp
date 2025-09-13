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

template <typename Iterator, typename T>
void ThreeWayPartition(Iterator first, Iterator last, const T& pivot_val,
                       Iterator& lt_end, Iterator& gt_begin) {
    Iterator lt = first;
    Iterator it = first;
    Iterator gt = last;

    while (it < gt) {
        if (*it < pivot_val) {
            std::iter_swap(lt, it);
            ++lt;
            ++it;
        } else if (pivot_val < *it) {
            --gt;
            std::iter_swap(it, gt);
        } else {
            ++it;
        }
    }
    lt_end = lt;
    gt_begin = gt;
}

template <typename Iterator>
void NthElement(Iterator first, Iterator nth, Iterator last) {
    using T = typename std::iterator_traits<Iterator>::value_type;

    while (first < last) {
        if (last - first <= 1) {
            return;
        }

        Iterator pivot = first + (last - first) / 2;
        T pivot_val = *pivot;

        Iterator lt_end;
        Iterator gt_begin;
        ThreeWayPartition(first, last, pivot_val, lt_end, gt_begin);

        if (nth < lt_end) {
            last = lt_end;
        } else if (nth >= gt_begin) {
            first = gt_begin;
        } else {
            return;
        }
    }
}

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

    NthElement(homes.begin(), homes.begin() + homes.size() / 2, homes.end());
    std::cout << LossFunction(homes, homes[homes.size() / 2]) << '\n';

    return 0;
}