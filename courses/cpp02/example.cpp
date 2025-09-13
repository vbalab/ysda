#include <cstdlib>
#include <iostream>
#include <optional>
#include <vector>
#include <memory>

// template <typename T>
// void f(T) = delete;

// template <typename T>
// void g() = delete;

template <typename T>
class SharedPtr {
    struct ControlBlock {
        T value;
        size_t count;

        template <typename... Args>
        ControlBlock(Args&&... args) : T(std::forward<Args>(args)...), count(0) {
        }
    };

    template <typename U, typename... Args>
    friend SharedPtr<U> MakeShared(Args&&... args);

    template <typename U>
    friend EnableSharedFromThis<U>;

public:
    SharedPtr(T* ptr) : ptr_(ptr), count_(new size_t(0)) {
        if constexpr (std::is_base_of_v<std::enable_shared_from_this<T>, T>) {
            ptr_->wptr = *this;
        }
    }

    ~SharedPtr() {
        if (count_ == nullptr) {
            return;
        }

        if (count_ && --*count_ == 0) {
            delete count_;
            delete ptr_;
        }
    }

    SharedPtr(const SharedPtr<T>& other) : ptr_(other.ptr_), count_(other.count_) {
        ++*count_;
    }

    SharedPtr& operator=(const SharedPtr<T>& other) {
        if (this != &other) {
            SharedPtr tmp(other);
            Swap(tmp);
        }

        return *this;
    }

    SharedPtr(SharedPtr<T>&& other) : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = nullptr;
    }

    SharedPtr& operator=(SharedPtr<T>&& other) {
        if (this != &other) {
            Swap(other);
        }

        return *this;
    }

    T& operator*() const {
        return *ptr_;
    }

    T* operator->() const {
        return ptr_;
    }

private:
    void Swap(SharedPtr& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(count_, other.count_);
    }

    SharedPtr(ControlBlock* block) : ptr_(&block->value), count_(&block->count) {
    }

private:
    T* ptr_;
    size_t* count_;
};

template <typename T, typename... Args>
SharedPtr<T> MakeShared(Args&&... args) {
    using Block = typename SharedPtr<T>::ControlBlock;

    Block* block = new Block(std::forward<Args>(args)...);
    return SharedPtr(block);
}

template <typename T>
struct EnableSharedFromThis {
    WeakPtr<T> wptr;

    SharedPtr<T> SharedFromThis() {
        return wptr.lock();
    }

    EnableSharedFromThis()
};

int main() {
    return 0;
}
