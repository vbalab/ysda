#include <iostream>
#include <vector>

struct Good {
    int64_t weight;
    int64_t value;

    friend std::ostream& operator<<(std::ostream& stream, const Good& good) {
        stream << good.weight << " " << good.value;
        return stream;
    }
};

struct Group {
    Good good;
    int64_t quantity;
};

std::vector<Good> IntoGoodsOfBinarySize(const std::vector<Group>& groups,
                                        int64_t max_weight) {
    std::vector<Good> goods;

    for (const Group& group : groups) {
        int64_t pow = 1;
        int64_t qty = group.quantity;

        while (pow < qty) {
            if (group.good.weight * pow > max_weight) {
                break;
            }
            goods.push_back(
                Good{group.good.weight * pow, group.good.value * pow});

            qty -= pow;
            pow <<= 1;
        }

        if (qty > 0 && group.good.weight * qty <= max_weight) {
            goods.push_back(
                Good{group.good.weight * qty, group.good.value * qty});
        }
    }

    return goods;
}

int64_t FindMaxValueBackpack(const std::vector<Good>& goods,
                             int64_t max_weight) {
    std::vector<int64_t> dp(max_weight + 1, 0);

    for (const Good& good : goods) {
        for (int64_t weight = max_weight; weight >= good.weight; --weight) {
            dp[weight] =
                std::max(dp[weight], dp[weight - good.weight] + good.value);
        }
    }

    int64_t max_value = 0;
    for (int64_t weight = 0; weight <= max_weight; ++weight) {
        if (dp[weight] > max_value) {
            max_value = dp[weight];
        }
    }

    return max_value;
}

std::vector<Group> InputGroups(size_t n_groups) {
    std::vector<Group> groups(n_groups);

    for (Group& group : groups) {
        std::cin >> group.good.weight >> group.good.value >> group.quantity;
    }

    return groups;
}

void Output(int64_t max_value) { std::cout << max_value << '\n'; }

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    size_t n_groups;
    int64_t max_weight;
    std::cin >> n_groups >> max_weight;

    std::vector<Group> groups = InputGroups(n_groups);
    std::vector<Good> goods = IntoGoodsOfBinarySize(groups, max_weight);

    int64_t max_value = FindMaxValueBackpack(goods, max_weight);
    Output(max_value);

    return 0;
}
