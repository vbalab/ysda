#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "a_maximum_in_sliding_window.cpp"

template <typename T>
void ExpectVectorsEqual(const std::vector<T>& result, const std::vector<T>& expected) {
    bool are_equal = std::equal(result.begin(), result.end(), expected.begin());

    if (!are_equal) {
        std::cout << "Test failed!\nResult:   ";
        for (const auto& val : result) {
            std::cout << val << " ";
        }
        std::cout << "\nExpected: ";
        for (const auto& val : expected) {
            std::cout << val << " ";
        }
        std::cout << '\n';
    }

    EXPECT_TRUE(are_equal) << "The arrays are not equal!";
}

TEST(MaxInSlidingWindowTest, SimpleTest) {
    std::vector<int32_t> v = { 3, 4, 1, 2 };
    std::vector<char> turns = { 'R', 'R', 'R', 'L', 'L', 'L' };

    std::vector<int32_t> result = MaxInSlidingWindow(v, turns);
    std::vector<int32_t> expected = { 4, 4, 4, 4, 2, 2 };

    ExpectVectorsEqual(result, expected);
}

TEST(MaxInSlidingWindowTest, YandexTest) {
    std::vector<int32_t> v = { 1, 4, 2, 3, 5, 8, 6, 7, 9, 10 };
    std::vector<char> turns = { 'R', 'R', 'L', 'R', 'R', 'R', 'L', 'L', 'L', 'R', 'L', 'L' };

    std::vector<int32_t> result = MaxInSlidingWindow(v, turns);
    std::vector<int32_t> expected = { 4, 4, 4, 4, 5, 8, 8, 8, 8, 8, 8, 6 };

    ExpectVectorsEqual(result, expected);
}

TEST(MaxInSlidingWindowTest, RandomInputTest) {
    std::vector<int32_t> v = { 9, 3, 8, 5, 2, 7 };
    std::vector<char> turns = { 'R', 'R', 'L', 'R', 'L' };

    std::vector<int32_t> result = MaxInSlidingWindow(v, turns);
    std::vector<int32_t> expected = { 9, 9, 8, 8, 8 };

    ExpectVectorsEqual(result, expected);
}

TEST(MaxInSlidingWindowTest, LargeInputTest) {
    std::size_t n = 100000;
    std::vector<int32_t> v(n, 1);
    v[0] = 100000;  // Make the first element the largest
    std::vector<char> turns(n-1, 'R');

    std::vector<int32_t> result = MaxInSlidingWindow(v, turns);
    std::vector<int32_t> expected(n-1, 100000);

    ExpectVectorsEqual(result, expected);
}

