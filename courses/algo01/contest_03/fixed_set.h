#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <ranges>
#include <vector>

static constexpr std::size_t kNumberRandomGen = 42;

template <typename T>
    requires std::integral<T>
class UniHash {
public:
    UniHash(int64_t slope, int64_t shift, int64_t prime)
        : slope_(slope), shift_(shift), prime_(prime) {}

    int64_t operator()(const T& value) const {
        int64_t result =
            (slope_ * static_cast<int64_t>(value) + shift_) % prime_;
        if (result < 0) {
            result += prime_;
        }

        return result;
    }

private:
    int64_t slope_;
    int64_t shift_;
    int64_t prime_;
};

template <typename T>
    requires std::integral<T>
class RandomUniHash : public UniHash<T> {
public:
    RandomUniHash() : UniHash<T>(GenRandom(), GenRandom(), kPrime) {}

private:
    static constexpr int64_t kPrime = 2000000011;

    static int64_t GenRandom() {
        static std::mt19937_64 gen{kNumberRandomGen};
        std::uniform_int_distribution<int64_t> distrib(1, kPrime - 1);
        return distrib(gen);
    }
};

template <typename T>
T CalculateSumOfSquares(const std::vector<T>& counts) {
    T sum_squares = 0;

    for (T count : counts) {
        sum_squares += count * count;
    }

    return sum_squares;
}

template <typename HashFunc>
HashFunc GenerateHash(std::function<HashFunc()> hash_generator,
                      std::function<bool(const HashFunc&)> predicate) {
    HashFunc hash;

    do {
        hash = hash_generator();
    } while (!predicate(hash));

    return hash;
}

template <typename BucketType, typename HashFunc>
std::vector<BucketType> DistributeByBuckets(
    const std::vector<int>& numbers, const HashFunc& hash_func,
    std::size_t n_buckets,
    const std::function<void(BucketType&, int)>& distribution_func) {
    std::vector<BucketType> buckets(n_buckets);

    for (const auto& num : numbers) {
        std::size_t hash = hash_func(num) % n_buckets;
        distribution_func(buckets[hash], num);
    }

    return buckets;
}

// perfect hashing
template <typename T>
    requires std::integral<T>
class SecondLevelHashTable {
public:
    SecondLevelHashTable() = default;

    void Initialize(const std::vector<T>& numbers) {
        std::size_t n_buckets = numbers.size() * numbers.size();

        buckets_.assign(n_buckets, std::nullopt);

        auto predicate = [&](const RandomUniHash<T>& hash_func) -> bool {
            auto distr = [](std::size_t& bucket, T) { ++bucket; };
            std::vector<std::size_t> bucket_counts =
                DistributeByBuckets<std::size_t, RandomUniHash<T>>(
                    numbers, hash_func, n_buckets, distr);

            return std::ranges::all_of(
                bucket_counts, [](const auto& count) { return count <= 1; });
        };

        hash_function_ = GenerateHash<RandomUniHash<T>>(
            [&]() { return RandomUniHash<T>(); }, predicate);

        auto distr = [](std::optional<T>& bucket, T num) { bucket = num; };
        buckets_ = DistributeByBuckets<std::optional<T>, RandomUniHash<T>>(
            numbers, *hash_function_, n_buckets, distr);
    }

    bool Contains(const T& number) const {
        if (buckets_.empty() || !hash_function_.has_value()) {
            return false;
        }

        std::size_t hash = hash_function_.value()(number) % buckets_.size();
        if (!buckets_[hash].has_value()) {
            return false;
        }

        return buckets_[hash].value() == number;
    }

private:
    std::optional<RandomUniHash<T>> hash_function_;
    std::vector<std::optional<T>> buckets_;
};

class FixedSet {
public:
    FixedSet() = default;

    void Initialize(const std::vector<int>& numbers) {
        if (numbers.empty()) {
            return;
        }

        std::size_t n_buckets =
            static_cast<std::size_t>(numbers.size() * kMultEnlarger);

        auto predicate = [&](const RandomUniHash<int>& hash_func) -> bool {
            auto distr = [](std::size_t& bucket, int) { ++bucket; };
            std::vector<std::size_t> bucket_counts =
                DistributeByBuckets<std::size_t, RandomUniHash<int>>(
                    numbers, hash_func, n_buckets, distr);

            std::size_t sum_squares = CalculateSumOfSquares(bucket_counts);

            return sum_squares <= n_buckets * kCompareDistributionParamK;
        };

        hash_function_ = GenerateHash<RandomUniHash<int>>(
            [&]() { return RandomUniHash<int>(); }, predicate);

        auto distr = [](std::vector<int>& bucket, int num) {
            bucket.push_back(num);
        };
        std::vector<std::vector<int>> bucket_contents =
            DistributeByBuckets<std::vector<int>, RandomUniHash<int>>(
                numbers, *hash_function_, n_buckets, distr);

        buckets_.clear();
        buckets_.resize(n_buckets);

        for (std::size_t i = 0; i < n_buckets; ++i) {
            if (!bucket_contents[i].empty()) {
                buckets_[i].Initialize(bucket_contents[i]);
            }
        }
    }

    bool Contains(int number) const {
        if (buckets_.empty() || !hash_function_.has_value()) {
            return false;
        }

        std::size_t hash = (*hash_function_)(number) % buckets_.size();

        return buckets_[hash].Contains(number);
    }

private:
    static constexpr double kMultEnlarger = 1.5;  // to get overload = 0.66
    static constexpr std::size_t kCompareDistributionParamK = 2;

    std::vector<SecondLevelHashTable<int>> buckets_;
    std::optional<RandomUniHash<int>> hash_function_;
};
