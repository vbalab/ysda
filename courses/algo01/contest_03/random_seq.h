#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

std::vector<int> UniqueRandomSequence(int size, int range) {
    // Ensure the requested size does not exceed the available range of numbers
    if (size > range) {
        throw std::invalid_argument("Size cannot be larger than the range of unique numbers available.");
    }
    
    // Initialize a vector with numbers from 0 to range - 1
    std::vector<int> sequence(range);
    std::iota(sequence.begin(), sequence.end(), 0);

    // Randomly shuffle the numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(sequence.begin(), sequence.end(), gen);

    // Return the first 'size' elements of the shuffled sequence
    return std::vector<int>(sequence.begin(), sequence.begin() + size);
}