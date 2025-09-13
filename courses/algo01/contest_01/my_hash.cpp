#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace std {
template <typename T1, typename T2>
struct hash<std::pair<T1, T2>> {
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t h1 = std::hash<T1>{}(p.first);
        std::size_t h2 = std::hash<T2>{}(p.second);

        return h1 ^ (h2 << 1);  // Simple hash combination
    }
};
}

template <typename keyType, typename valueType>
class MyChainingHashMap {
   private:
    int size;
    float load_factor;
    std::vector<std::vector<std::pair<keyType, valueType>>> buckets;

   public:
    MyChainingHashMap(const int n_buckets) : size(0), load_factor(0.75), buckets(n_buckets) {
    }

    std::size_t hash(const keyType& key) {
        return std::hash<keyType>{}(key) % this->buckets.size();
    }

    typename std::vector<std::pair<keyType, valueType>>::iterator find(const keyType& key) {
        auto& bucket = this->buckets[this->hash(key)];
        return std::find_if(bucket.begin(), bucket.end(),
                            [&](const auto& kv) { return kv.first == key; });
    }

    typename std::vector<std::pair<keyType, valueType>>::iterator end() {
        static std::vector<std::pair<keyType, valueType>> empty_bucket;
        return empty_bucket.end();
    }

    valueType& operator[](const keyType& key) {
        std::size_t bucket_index = this->hash(key);
        auto& bucket = this->buckets[bucket_index];

        for (auto& kv: bucket) {
            if (kv.first == key) {
                return kv.second;
            }
        }

        bucket.emplace_back(key, valueType());
        ++this->size;
        return bucket.back().second;
    }

    void insert(const keyType& key, const valueType& value) {
        this->operator[](key) = value;
    }
};

int main() {
    MyChainingHashMap<std::pair<int, bool>, std::vector<int>> my_map{20};

    my_map.insert({1, true}, {10, 20, 30});
    my_map[{2, false}] = {40, 50};
    my_map[{1, false}] = {60, 70};

    auto vec1 = my_map[{1, true}];
    auto vec2 = my_map[{2, false}];

    std::cout << "Key {1, true}: ";
    for (int v: vec1) {
        std::cout << v << " ";  // Output: 10 20 30
    }
    std::cout << '\n';

    std::cout << "Key {2, false}: ";
    for (int v: vec2) {
        std::cout << v << " ";  // Output: 40 50
    }
    std::cout << '\n';

    if (my_map.find({1, true}) != my_map.end()) {
        std::cout << "Ok!" << '\n';
    }

    if (my_map.find({10, true}) == my_map.end()) {
        std::cout << "Ok!" << '\n';
    }

    return 0;
}