#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

struct Element {
    int64_t value;
    int64_t index_in_heap;
};

class PriorityQueue {
public:
    PriorityQueue(bool is_max_heap) : is_max_heap_(is_max_heap) {}

    void Insert(std::vector<Element>::iterator element) {
        element->index_in_heap = heap_elements_.size();
        heap_elements_.push_back(element);

        HeapifyUp(element);
    }

    auto RemoveTop() {
        int64_t last_index = heap_elements_.size() - 1;

        SwapElements(0, last_index);

        auto top_element = heap_elements_.back();
        heap_elements_.pop_back();

        if (!heap_elements_.empty()) {
            HeapifyDown(heap_elements_.front());
        }

        return top_element;
    }

    void Remove(std::vector<Element>::iterator it) {
        int64_t idx = it->index_in_heap;

        int64_t last_index = heap_elements_.size() - 1;

        SwapElements(idx, last_index);
        heap_elements_.pop_back();

        if (idx < static_cast<int64_t>(heap_elements_.size())) {
            HeapifyUp(heap_elements_[idx]);
            HeapifyDown(heap_elements_[idx]);
        }
    }

    bool IsEmpty() const { return heap_elements_.empty(); }

    auto Top() const { return heap_elements_.front(); }

    auto Size() const { return heap_elements_.size(); }

    std::vector<Element>::iterator& operator[](int64_t idx) {
        return heap_elements_[idx];
    }

private:
    std::vector<std::vector<Element>::iterator> heap_elements_;
    bool is_max_heap_;

    bool Compare(std::vector<Element>::iterator a_par,
                 std::vector<Element>::iterator b_par) const {
        return is_max_heap_ ? (a_par->value < b_par->value)
                            : (a_par->value > b_par->value);
    }

    void HeapifyUp(std::vector<Element>::iterator pos) {
        while ((pos->index_in_heap > 0) &&
               Compare(heap_elements_[(pos->index_in_heap - 1) >> 1], pos)) {
            SwapElements(pos->index_in_heap, (pos->index_in_heap - 1) >> 1);
        }
    }

    void HeapifyDown(std::vector<Element>::iterator pos) {
        while (((pos->index_in_heap << 1) + 1) <
               static_cast<int64_t>(heap_elements_.size())) {
            int64_t left_child = (pos->index_in_heap << 1) + 1;
            int64_t right_child = (pos->index_in_heap << 1) + 2;

            int64_t selected_child = left_child;
            if ((right_child < static_cast<int64_t>(heap_elements_.size())) &&
                Compare(heap_elements_[left_child],
                        heap_elements_[right_child])) {
                selected_child = right_child;
            }

            if (Compare(heap_elements_[selected_child], pos)) {
                break;
            }

            SwapElements(pos->index_in_heap, selected_child);
        }
    }

    void SwapElements(int64_t idx_a, int64_t idx_b) {
        auto temp = heap_elements_[idx_a];

        heap_elements_[idx_a] = heap_elements_[idx_b];
        heap_elements_[idx_b] = temp;

        auto temp_index = heap_elements_[idx_a]->index_in_heap;

        heap_elements_[idx_a]->index_in_heap =
            heap_elements_[idx_b]->index_in_heap;

        heap_elements_[idx_b]->index_in_heap = temp_index;
    }
};

class SlidingWindowManager {
public:
    SlidingWindowManager(int64_t size, int64_t k_par)
        : k_position_(k_par),
          max_heap_(true),
          min_heap_(false),
          in_max_heap_(std::vector<bool>(size, false)),
          elements_(std::vector<Element>(size)) {
        left_ptr_ = elements_.begin();
        right_ptr_ = elements_.begin();
    }

    void Add(int64_t value, int64_t index) {
        elements_[index].value = value;

        if (max_heap_.IsEmpty()) {
            max_heap_.Insert(right_ptr_);
            in_max_heap_[index] = true;
        }
    }

    void MoveRight() {
        ++right_ptr_;
        if (static_cast<int64_t>(max_heap_.Size()) < k_position_) {
            max_heap_.Insert(right_ptr_);

            in_max_heap_[right_ptr_ - elements_.begin()] = true;
        } else if (!max_heap_.IsEmpty() &&
                   right_ptr_->value < max_heap_.Top()->value) {
            auto top_element = max_heap_.RemoveTop();

            in_max_heap_[top_element - elements_.begin()] = false;

            min_heap_.Insert(top_element);
            max_heap_.Insert(right_ptr_);

            in_max_heap_[right_ptr_ - elements_.begin()] = true;
        } else {
            min_heap_.Insert(right_ptr_);
        }

        if (static_cast<int64_t>(max_heap_.Size()) == k_position_) {
            std::cout << max_heap_.Top()->value << '\n';
        } else {
            std::cout << "-1" << '\n';
        }
    }

    void MoveLeft() {
        if (in_max_heap_[left_ptr_ - elements_.begin()]) {
            max_heap_.Remove(left_ptr_);

            if (!min_heap_.IsEmpty() &&
                (static_cast<int64_t>(max_heap_.Size()) < k_position_)) {
                auto top_element = min_heap_.RemoveTop();

                max_heap_.Insert(top_element);

                in_max_heap_[top_element - elements_.begin()] = true;
            }
        } else if (!min_heap_.IsEmpty()) {
            min_heap_.Remove(left_ptr_);
        }

        ++left_ptr_;

        if (static_cast<int64_t>(max_heap_.Size()) == k_position_) {
            std::cout << max_heap_.Top()->value << '\n';
        } else {
            std::cout << "-1" << '\n';
        }
    }

private:
    int64_t k_position_;
    PriorityQueue max_heap_;
    PriorityQueue min_heap_;
    std::vector<bool> in_max_heap_;
    std::vector<Element> elements_;
    std::vector<Element>::iterator left_ptr_, right_ptr_;
};

int main() {
    int64_t size;
    int64_t queries;
    int64_t k_par;

    std::cin >> size >> queries >> k_par;

    SlidingWindowManager manager(size, k_par);

    int64_t value;

    for (int64_t i = 0; i < size; ++i) {
        std::cin >> value;
        manager.Add(value, i);
    }

    std::string actions;
    std::cin >> actions;

    for (int64_t i = 0; i < queries; ++i) {
        if (actions[i] == 'R') {
            manager.MoveRight();
        } else if (actions[i] == 'L') {
            manager.MoveLeft();
        }
    }

    return 0;
}
