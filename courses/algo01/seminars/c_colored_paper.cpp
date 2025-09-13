#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

static constexpr std::size_t kMaxColors = 3000;

struct Rectangle {
    int64_t left;
    int64_t bottom;
    int64_t right;
    int64_t top;

    int64_t deep;
    int64_t color;

    friend std::ostream& operator<<(std::ostream& os, const Rectangle& rect) {
        os << "Rectangle("
           << "left: " << rect.left << ", "
           << "bottom: " << rect.bottom << ", "
           << "right: " << rect.right << ", "
           << "top: " << rect.top << ", "
           << "deep: " << rect.deep << ", "
           << "color: " << rect.color << ")";
        return os;
    }
};

template <typename T>
class Heap {
public:
    Heap(std::function<bool(const T&, const T&)> cmp) : cmp_(cmp) {}

    void Build(const std::vector<T>& elements) {
        data_ = elements;
        std::make_heap(data_.begin(), data_.end(), cmp_);
    }

    void Push(const T& value) {
        data_.push_back(value);
        std::push_heap(data_.begin(), data_.end(), cmp_);
    }

    void Pop() {
        if (!data_.empty()) {
            std::pop_heap(data_.begin(), data_.end(), cmp_);
            data_.pop_back();
        }
    }

    const T& Top() const {
        if (data_.empty()) {
            throw std::runtime_error("Heap is empty");
        }
        return data_.front();
    }

    bool Empty() const { return data_.empty(); }

    size_t Size() const { return data_.size(); }

private:
    std::vector<T> data_;
    std::function<bool(const T&, const T&)> cmp_;
};

bool VerticalComparator(const Rectangle& left, const Rectangle& right) {
    return left.bottom < right.bottom;
};

bool DeepComparator(const Rectangle& left, const Rectangle& right) {
    return left.deep < right.deep;
};

bool HorizontalComparator(int64_t left, int64_t right) { return left < right; };

bool WithinVertical(const Rectangle& rect, int64_t left, int64_t right) {
    return (rect.left <= left) && (rect.right >= right);
}

void ScanLine(const std::vector<Rectangle>& rects, std::vector<int64_t>& colors,
              int64_t left, int64_t right) {
    int64_t width = right - left;

    Heap<Rectangle> heap(DeepComparator);
    heap.Push(rects[0]);

    int64_t start_cut;
    int64_t end_cut = 0;
    for (std::size_t i = 1; i < rects.size() + 1; ++i) {
        start_cut = end_cut;

        while (heap.Top().top <= start_cut) {
            heap.Pop();
        }

        if ((i < rects.size()) && (heap.Top().top >= rects[i].bottom)) {
            end_cut = rects[i].bottom;
            colors[heap.Top().color] += width * (end_cut - start_cut);

            if (WithinVertical(rects[i], left, right)) {
                heap.Push(rects[i]);
            }
        } else {
            end_cut = heap.Top().top;
            colors[heap.Top().color] += width * (end_cut - start_cut);
        }
    }

    if (end_cut != rects[0].top) {
        colors[rects[0].color] += width * (end_cut - start_cut);
    }
}

std::vector<int64_t> FindColorDistribution(std::vector<Rectangle>& rects) {
    std::vector<int64_t> areas;

    for (const Rectangle& rect : rects) {
        areas.push_back(rect.left);
        areas.push_back(rect.right);
    }

    std::sort(areas.begin(), areas.end(), HorizontalComparator);

    std::sort(rects.begin(), rects.end(), VerticalComparator);

    std::vector<int64_t> colors(kMaxColors, 0);
    for (std::size_t i = 0; i < areas.size() - 1; ++i) {
        ScanLine(rects, colors, areas[i], areas[i + 1]);
    }

    return colors;
}

std::vector<Rectangle> Input() {
    int64_t width;
    int64_t length;
    std::size_t n_rects;
    std::cin >> width >> length >> n_rects;

    std::vector<Rectangle> rects(n_rects + 1);
    rects[0] = Rectangle(0, 0, width, length, 0, 1);

    for (std::size_t i = 1; i < n_rects + 1; ++i) {
        rects[i].deep = i;

        std::cin >> rects[i].left >> rects[i].bottom >> rects[i].right >>
            rects[i].top >> rects[i].color;
    }

    return rects;
}

void Output(std::vector<int64_t>& colors) {
    for (std::size_t i = 0; i < colors.size(); ++i) {
        if (colors[i] > 0) {
            std::cout << i << ' ' << colors[i] << '\n';
        }
    }
}

int main() {
    std::vector<Rectangle> rects = Input();

    std::vector<int64_t> colors = FindColorDistribution(rects);

    Output(colors);

    return 0;
}