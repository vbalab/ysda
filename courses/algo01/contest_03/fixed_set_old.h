#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

template <typename T>
class UniversalFamily {
public:
    UniversalFamily()
        : distrib_(1, INT32_MAX), a_(distrib_(gen_)), b_(distrib_(gen_)) {}

    std::size_t operator()(const T& value) const {
        return static_cast<std::size_t>(a_ * value + b_);
    }

private:
    std::mt19937 gen_{std::random_device{}()};
    std::uniform_int_distribution<std::size_t> distrib_;
    std::size_t a_;
    std::size_t b_;
};

template <typename T>
class ChainingSet {
public:
    ChainingSet() {}

    ChainingSet(const std::vector<T>& values)
        : distrib_(0, kAddEnlarger),
          n_buckets_(static_cast<std::size_t>(values.size() * kMultEnlarger) +
                     distrib_(gen_)),
          buckets_(n_buckets_),
          hash_(UniversalFamily<T>()) {
        for (const T& val : values) {
            buckets_[GetHash(val)].push_back(val);
        }
    }

    std::size_t GetHash(const T& value) const {
        return hash_(value) % n_buckets_;
    }

    bool Contains(const T& value) const {
        return std::any_of(buckets_[GetHash(value)].begin(),
                           buckets_[GetHash(value)].end(),
                           [&value](const T& val) { return val == value; });
    }

private:
    std::mt19937 gen_{std::random_device{}()};
    std::uniform_int_distribution<std::size_t> distrib_;
    static constexpr double kMultEnlarger = 1.5;
    static constexpr double kAddEnlarger = 1024;

    int32_t n_buckets_;
    std::vector<std::vector<T>> buckets_;
    UniversalFamily<T> hash_;
};

class FixedSet {
public:
    FixedSet() {}

    void Initialize(const std::vector<int>& numbers) {
        set_ = ChainingSet(numbers);
    }

    bool Contains(int number) const { return set_.Contains(number); }

private:
    ChainingSet<int> set_;
};
