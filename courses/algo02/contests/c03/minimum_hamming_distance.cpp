#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

const int kInf = 1000000000;

class TransformationDP {
public:
    TransformationDP(const std::string& alpha, const std::string& beta,
                     int max_edits)
        : alpha_(alpha),
          beta_(beta),
          kMaxEdits(max_edits),
          alpha_length_(static_cast<int>(alpha.size())),
          beta_length_(static_cast<int>(beta.size())),
          offset_(max_edits),
          width_val_(2 * max_edits + 1),
          dp_table_(max_edits + 1,
                    std::vector<int>((alpha_length_ + 1) * width_val_, kInf)) {
        dp_table_[0][GetDPIndex(0, 0)] = 0;
    }

    int Compute() {
        if (std::abs(alpha_length_ - beta_length_) > kMaxEdits) {
            return -1;
        }

        for (int edits = 0; edits <= kMaxEdits; ++edits) {
            for (int i_idx = 0; i_idx <= alpha_length_; ++i_idx) {
                for (int d_val = -kMaxEdits; d_val <= kMaxEdits; ++d_val) {
                    int j_idx = i_idx + d_val;
                    if (j_idx < 0 || j_idx > beta_length_) {
                        continue;
                    }
                    int current_cost =
                        dp_table_[edits][GetDPIndex(i_idx, d_val)];
                    if (current_cost == kInf) {
                        continue;
                    }
                    ProcessCopy(i_idx, d_val, edits);
                    ProcessSubstitution(i_idx, d_val, edits);
                    ProcessInsertion(i_idx, d_val, edits);
                    ProcessDeletion(i_idx, d_val, edits);
                }
            }
        }

        int target_d = beta_length_ - alpha_length_;
        if (target_d < -kMaxEdits || target_d > kMaxEdits) {
            return -1;
        }
        int min_cost = kInf;
        for (int edits = 0; edits <= kMaxEdits; ++edits) {
            min_cost =
                std::min(min_cost,
                         dp_table_[edits][GetDPIndex(alpha_length_, target_d)]);
        }
        return (min_cost == kInf ? -1 : min_cost);
    }

private:
    inline int GetDPIndex(int i_idx, int d_val) const {
        return i_idx * width_val_ + (d_val + offset_);
    }

    void ProcessCopy(int i_idx, int d_val, int edits) {
        int j_idx = i_idx + d_val;
        if (i_idx < alpha_length_ && j_idx < beta_length_) {
            int current_cost = dp_table_[edits][GetDPIndex(i_idx, d_val)];
            int additional_cost = (alpha_[i_idx] == beta_[j_idx]) ? 0 : 1;
            int new_cost = current_cost + additional_cost;
            dp_table_[edits][GetDPIndex(i_idx + 1, d_val)] = std::min(
                dp_table_[edits][GetDPIndex(i_idx + 1, d_val)], new_cost);
        }
    }

    void ProcessSubstitution(int i_idx, int d_val, int edits) {
        int j_idx = i_idx + d_val;
        if (i_idx < alpha_length_ && j_idx < beta_length_ &&
            edits < kMaxEdits) {
            int current_cost = dp_table_[edits][GetDPIndex(i_idx, d_val)];
            int new_cost = current_cost;
            dp_table_[edits + 1][GetDPIndex(i_idx + 1, d_val)] = std::min(
                dp_table_[edits + 1][GetDPIndex(i_idx + 1, d_val)], new_cost);
        }
    }

    void ProcessInsertion(int i_idx, int d_val, int edits) {
        int j_idx = i_idx + d_val;
        if (j_idx < beta_length_ && edits < kMaxEdits && d_val < kMaxEdits) {
            int current_cost = dp_table_[edits][GetDPIndex(i_idx, d_val)];
            int new_cost = current_cost;
            dp_table_[edits + 1][GetDPIndex(i_idx, d_val + 1)] = std::min(
                dp_table_[edits + 1][GetDPIndex(i_idx, d_val + 1)], new_cost);
        }
    }

    void ProcessDeletion(int i_idx, int d_val, int edits) {
        if (i_idx < alpha_length_ && edits < kMaxEdits && d_val > -kMaxEdits) {
            int current_cost = dp_table_[edits][GetDPIndex(i_idx, d_val)];
            int new_cost = current_cost;
            dp_table_[edits + 1][GetDPIndex(i_idx + 1, d_val - 1)] =
                std::min(dp_table_[edits + 1][GetDPIndex(i_idx + 1, d_val - 1)],
                         new_cost);
        }
    }

    const std::string& alpha_;
    const std::string& beta_;
    const int kMaxEdits;
    int alpha_length_;
    int beta_length_;
    int offset_;
    int width_val_;
    std::vector<std::vector<int>> dp_table_;
};

int ComputeMinHamming(const std::string& alpha, const std::string& beta,
                      int max_edits) {
    TransformationDP dp_solver(alpha, beta, max_edits);
    return dp_solver.Compute();
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string alpha;
    std::string beta;
    int max_edits;

    std::cin >> alpha >> beta >> max_edits;

    std::cout << ComputeMinHamming(alpha, beta, max_edits) << "\n";

    return 0;
}
