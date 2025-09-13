#include <iostream>
#include <vector>

struct Subsequence {
    int start_index;
    std::vector<int> seq;

    Subsequence() : start_index(-1) {
    }

    bool operator>(const Subsequence& other) const {
        if (seq.size() != other.seq.size())
            return seq.size() > other.seq.size();

        return start_index < other.start_index;
    }
};

class LongestAlteringSubsequence {
   private:
    std::vector<int> v;
    std::vector<std::vector<Subsequence>> m;
    std::vector<std::vector<bool>> computed;

   public:
    std::vector<int> calculate(const std::vector<int>& v) {
        this->v = v;
        std::size_t n = v.size();

        m.assign(n, std::vector<Subsequence>(2));
        computed.assign(n, std::vector<bool>(2, false));

        Subsequence max;
        Subsequence cur;

        for (std::size_t i = 0; i < n; ++i) {
            cur = helper(true, i);
            cur.seq.insert(cur.seq.begin(), this->v[i]);
            cur.start_index = i;
            if (cur > max) {
                max = cur;
            }

            cur = helper(false, i);
            cur.seq.insert(cur.seq.begin(), this->v[i]);
            cur.start_index = i;
            if (cur > max) {
                max = cur;
            }
        }

        return max.seq;
    }

    Subsequence helper(bool less, std::size_t index) {
        int less_index = less ? 1 : 0;

        if (computed[index][less_index]) {
            return m[index][less_index];
        }

        int previous = this->v[index];

        Subsequence max;
        Subsequence cur;

        for (std::size_t i = index + 1; i < this->v.size(); ++i) {
            if ((less && previous > this->v[i]) || (!less && previous < this->v[i])) {
                cur = helper(!less, i);
                cur.seq.insert(cur.seq.begin(), this->v[i]);
                cur.start_index = i;

                if (cur > max) {
                    max = cur;
                }
            }
        }

        m[index][less_index] = max;
        computed[index][less_index] = true;

        return max;
    }
};

int main() {
    std::size_t n;
    std::cin >> n;

    std::vector<int> v(n);
    for (int& num: v) {
        std::cin >> num;
    }

    LongestAlteringSubsequence las;
    std::vector<int> result = las.calculate(v);

    for (const int& num: result) {
        std::cout << num << ' ';
    }
    std::cout << '\n';

    return 0;
}
