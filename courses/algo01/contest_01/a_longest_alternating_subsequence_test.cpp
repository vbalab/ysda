#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "a_longest_alternating_subsequence.cpp"  // Include your actual solution file

// Function to invoke your solution
std::vector<int> longestAlternatingSubsequence(const std::vector<int>& v) {
    LongestAlteringSubsequence las;  // Use your class
    return las.calculate(v);  // Call your method
}

// Test Case 1: Simple Alternating Sequence
TEST(LongestAlternatingSubsequenceTest, SimpleAlternatingSequence) {
    std::vector<int> input = {1, 4, 2, 3, 5, 8, 6, 7, 9, 10};
    std::vector<int> expected_output = {1, 4, 2, 8, 6, 7};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 2: Strictly Increasing Sequence
TEST(LongestAlternatingSubsequenceTest, StrictlyIncreasingSequence) {
    std::vector<int> input = {1, 2, 3, 4, 5};
    std::vector<int> expected_output = {1, 2};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 3: Single Element
TEST(LongestAlternatingSubsequenceTest, SingleElement) {
    std::vector<int> input = {100};
    std::vector<int> expected_output = {100};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 4: Strictly Decreasing Sequence
TEST(LongestAlternatingSubsequenceTest, StrictlyDecreasingSequence) {
    std::vector<int> input = {6, 5, 4, 3, 2, 1};
    std::vector<int> expected_output = {6, 5};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 5: All Elements Same
TEST(LongestAlternatingSubsequenceTest, AllElementsSame) {
    std::vector<int> input = {7, 7, 7, 7};
    std::vector<int> expected_output = {7};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 6: Mixed Increasing and Decreasing
TEST(LongestAlternatingSubsequenceTest, MixedIncreasingDecreasing) {
    std::vector<int> input = {3, 5, 4, 6, 3, 7, 2, 8};
    std::vector<int> expected_output = {3, 5, 4, 6, 3, 7, 2, 8};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 7: Short Sequence
TEST(LongestAlternatingSubsequenceTest, ShortSequence) {
    std::vector<int> input = {1, 2};
    std::vector<int> expected_output = {1, 2};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 8: Two Decreasing Elements
TEST(LongestAlternatingSubsequenceTest, TwoDecreasingElements) {
    std::vector<int> input = {2, 1};
    std::vector<int> expected_output = {2, 1};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 9: Alternating Peak Sequence
TEST(LongestAlternatingSubsequenceTest, AlternatingPeakSequence) {
    std::vector<int> input = {1, 3, 2, 4, 3, 5};
    std::vector<int> expected_output = {1, 3, 2, 4, 3, 5};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 10: Large Sequence with Repeats
TEST(LongestAlternatingSubsequenceTest, LargeSequenceWithRepeats) {
    std::vector<int> input = {2, 2, 2, 3, 3, 3, 1};
    std::vector<int> expected_output = {2, 3, 1};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 11: Alternating Peak Sequence with All Values Being Negative
TEST(LongestAlternatingSubsequenceTest, AlternatingNegativeNumbers) {
    std::vector<int> input = {-1, -3, -2, -4, -3, -5};
    std::vector<int> expected_output = {-1, -3, -2, -4, -3, -5};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 12: Long Sequence of Repeated Values and Alternating Peaks
TEST(LongestAlternatingSubsequenceTest, RepeatedValuesAndAlternatingPeaks) {
    std::vector<int> input = {2, 2, 2, 3, 2, 3, 2, 3};
    std::vector<int> expected_output = {2, 3, 2, 3, 2, 3};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 13: Edge Case with the Minimum and Maximum Possible Values
TEST(LongestAlternatingSubsequenceTest, ExtremeValues) {
    std::vector<int> input = {-1000000000, 1000000000, -1000000000, 1000000000};
    std::vector<int> expected_output = {-1000000000, 1000000000, -1000000000, 1000000000};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 14: Sequence with Only Two Alternating Elements
TEST(LongestAlternatingSubsequenceTest, TwoAlternatingElements) {
    std::vector<int> input = {1, 2, 1, 2, 1, 2};
    std::vector<int> expected_output = {1, 2, 1, 2, 1, 2};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Test Case 15: Very Large Sequence with Simple Alternation
TEST(LongestAlternatingSubsequenceTest, LargeAlternatingSequence) {
    std::vector<int> input(1000);
    for (int i = 0; i < 1000; i++) {
        input[i] = (i % 2 == 0) ? i : -i;
    }
    ASSERT_EQ(longestAlternatingSubsequence(input).size(), 1000);
}

// Test Case 16: Sequence with Large Values but Small Changes
TEST(LongestAlternatingSubsequenceTest, LargeValuesSmallDifferences) {
    std::vector<int> input = {1000000000, 999999999, 1000000000, 999999999};
    std::vector<int> expected_output = {1000000000, 999999999, 1000000000, 999999999};
    ASSERT_EQ(longestAlternatingSubsequence(input), expected_output);
}

// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
