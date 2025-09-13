#include <random>

template <typename T>
T GenerateRandomT() {
    std::random_device rd;
    std::mt19937 generator(rd());

    std::uniform_int_distribution<T> distribution(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    return distribution(generator);
}
