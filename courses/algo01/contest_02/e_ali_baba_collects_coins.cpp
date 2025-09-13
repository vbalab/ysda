#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

struct Coin {
    uint64_t pos;
    uint64_t time;
};

void UpdateDpLeft(std::size_t left, std::size_t right,
                  const std::vector<Coin>& coins,
                  std::vector<std::vector<std::vector<uint64_t>>>& dp) {
    if (dp[left + 1][right][0] != UINT64_MAX) {
        uint64_t time =
            dp[left + 1][right][0] + coins[left + 1].pos - coins[left].pos;
        if (time <= coins[left].time) {
            dp[left][right][0] = std::min(dp[left][right][0], time);
        }
    }
    if (dp[left + 1][right][1] != UINT64_MAX) {
        uint64_t time =
            dp[left + 1][right][1] + coins[right].pos - coins[left].pos;
        if (time <= coins[left].time) {
            dp[left][right][0] = std::min(dp[left][right][0], time);
        }
    }
}

void UpdateDpRight(std::size_t left, std::size_t right,
                   const std::vector<Coin>& coins,
                   std::vector<std::vector<std::vector<uint64_t>>>& dp) {
    if (dp[left][right - 1][0] != UINT64_MAX) {
        uint64_t time =
            dp[left][right - 1][0] + coins[right].pos - coins[left].pos;
        if (time <= coins[right].time) {
            dp[left][right][1] = std::min(dp[left][right][1], time);
        }
    }
    if (dp[left][right - 1][1] != UINT64_MAX) {
        uint64_t time =
            dp[left][right - 1][1] + coins[right].pos - coins[right - 1].pos;
        if (time <= coins[right].time) {
            dp[left][right][1] = std::min(dp[left][right][1], time);
        }
    }
}

void CollectCoins(const std::vector<Coin>& coins, std::size_t n_coins) {
    std::vector<std::vector<std::vector<uint64_t>>> dp(
        n_coins, std::vector<std::vector<uint64_t>>(
                     n_coins, std::vector<uint64_t>(2, UINT64_MAX)));

    for (std::size_t i = 0; i < n_coins; ++i) {
        dp[i][i][0] = 0;
        dp[i][i][1] = 0;
    }

    for (std::size_t len = 1; len < n_coins; ++len) {
        for (std::size_t left = 0; left + len < n_coins; ++left) {
            std::size_t right = left + len;

            UpdateDpLeft(left, right, coins, dp);
            UpdateDpRight(left, right, coins, dp);
        }
    }

    uint64_t result = std::min(dp[0][n_coins - 1][0], dp[0][n_coins - 1][1]);

    if (result == UINT64_MAX) {
        std::cout << "No solution" << std::endl;
    } else {
        std::cout << result << std::endl;
    }
}

int main() {
    std::size_t n_coins;
    std::cin >> n_coins;

    std::vector<Coin> coins(n_coins);
    for (Coin& coin : coins) {
        std::cin >> coin.pos >> coin.time;
    }

    std::sort(coins.begin(), coins.end(),
              [](const Coin& coin_a, const Coin& coin_b) {
                  return coin_a.pos < coin_b.pos;
              });

    CollectCoins(coins, n_coins);

    return 0;
}
