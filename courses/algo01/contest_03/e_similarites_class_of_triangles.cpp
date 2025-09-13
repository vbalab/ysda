#include <cstdint>
#include <iostream>
#include <vector>

struct Triangle {
    int64_t side1, side2, side3;
};

inline void Swap(int64_t& num1, int64_t& num2) {
    if (num1 != num2) {
        num1 ^= num2;
        num2 ^= num1;
        num1 ^= num2;
    }
}

int64_t ComputeGCD(int64_t a_par, int64_t b_par) {
    if (b_par > a_par) {
        Swap(a_par, b_par);
    }
    int64_t temp;
    while (b_par > 0) {
        temp = a_par % b_par;
        a_par = b_par;
        b_par = temp;
    }
    return a_par;
}

const uint64_t kMOD = 1000000000 + 7;

class TriangleHashMap {
public:
    TriangleHashMap(int64_t size)
        : size_(size),
          count_(0),
          triangles_(std::vector<std::vector<Triangle>>(size)) {}

    void Insert(Triangle new_triangle) {
        uint64_t hash_value = ComputeHash(new_triangle) % size_;
        int64_t bucket_size = triangles_[hash_value].size();

        bool is_new = true;

        for (int64_t i = 0; i < bucket_size; ++i) {
            if (AreEqual(triangles_[hash_value][i], new_triangle)) {
                is_new = false;

                break;
            }
        }

        if (is_new) {
            ++count_;
            triangles_[hash_value].push_back(new_triangle);
        }
    }

    int64_t Count() const { return count_; }

private:
    int64_t size_;
    int64_t count_;

    std::vector<std::vector<Triangle>> triangles_;

    std::vector<uint64_t> hash_primes_ = {
        kMOD, (kMOD * kMOD) % UINT64_MAX,
        (kMOD * kMOD % UINT64_MAX * kMOD) % UINT64_MAX};

    uint64_t ComputeHash(const Triangle& triangle) {
        return triangle.side1 * hash_primes_[0] +
               triangle.side2 * hash_primes_[1] +
               triangle.side3 * hash_primes_[2];
    }

    static bool AreEqual(const Triangle& t1, const Triangle& t2) {
        return (t1.side1 == t2.side1) && (t1.side2 == t2.side2) &&
               (t1.side3 == t2.side3);
    }
};

void Sort(std::vector<int64_t>& sides) {
    for (int i = 0; i < 2; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            if (sides[i] > sides[j]) {
                Swap(sides[i], sides[j]);
            }
        }
    }
}

int main() {
    int64_t n_triangles;
    std::cin >> n_triangles;

    TriangleHashMap hash_map(n_triangles);

    std::vector<int64_t> sides(3);
    for (int64_t i = 0; i < n_triangles; ++i) {
        int64_t gcd_value;

        for (int side_index = 0; side_index < 3; ++side_index) {
            std::cin >> sides[side_index];

            if (side_index == 0) {
                gcd_value = sides[side_index];
            } else {
                gcd_value = ComputeGCD(gcd_value, sides[side_index]);
            }
        }

        Sort(sides);

        if (gcd_value == 0) {
            hash_map.Insert({sides[0], sides[1], sides[2]});
        } else {
            hash_map.Insert({sides[0] / gcd_value, sides[1] / gcd_value,
                             sides[2] / gcd_value});
        }
    }

    std::cout << hash_map.Count() << '\n';

    return 0;
}
