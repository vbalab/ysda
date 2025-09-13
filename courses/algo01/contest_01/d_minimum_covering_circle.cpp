#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

std::vector<std::pair<double, bool>> calculate_segments(
    const std::vector<std::pair<int, int>>& points, double r) {
    // true == "opens", false == "closes"
    std::vector<std::pair<double, bool>> segments;

    double diff;
    for (const std::pair<int, int>& point: points) {
        if (std::fabs(point.second) <= r) {
            diff = std::sqrtl(r * r - point.second * point.second);
            segments.push_back({point.first - diff, true});
            segments.push_back({point.first + diff, false});
        }
    }

    return segments;
}

int covered(const std::vector<std::pair<int, int>>& points, double r) {
    std::vector<std::pair<double, bool>> segments = calculate_segments(points, r);

    std::sort(segments.begin(), segments.end(),
              [](const std::pair<double, bool>& a, const std::pair<double, bool>& b) {
                  return a.first < b.first;
              });

    int max = 0;
    int cur = 0;
    for (const std::pair<double, bool>& c: segments) {
        if (c.second) {
            ++cur;
        } else {
            --cur;
        }

        max = std::max(cur, max);
    }

    return max;
}

double binary_search_on_condition(const std::vector<std::pair<int, int>>& points, int k, double l,
                                  double u) {
    double m = (l + u) / 2;

    if (u - l < 1e-7) {
        return m;
    }

    int n_covered = covered(points, m);

    if (n_covered < k) {
        return binary_search_on_condition(points, k, m, u);
    } else {
        return binary_search_on_condition(points, k, l, m);
    }
}

int main() {
    int n;
    int k;
    std::cin >> n >> k;

    std::vector<std::pair<int, int>> points(n);
    for (std::pair<int, int>& num: points) {
        std::cin >> num.first >> num.second;
    }

    std::sort(points.begin(), points.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  return a.first < b.first;
              });

    std::cout << std::fixed << std::setprecision(6);
    std::cout << binary_search_on_condition(points, k, 0., 1500.) << '\n';

    return 0;
}
