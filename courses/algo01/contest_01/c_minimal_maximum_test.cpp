#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "c_minimal_maximum.cpp"  // Include your actual solution file

// Test case 1: Basic small arrays
TEST(FindMinMaxTest, BasicSmallArrays) {
    std::vector<int> A = {1, 1, 1};
    std::vector<int> B = {1, 1, 1};
    EXPECT_EQ(find_min_max(A, B, 0, 2), 1);
}

