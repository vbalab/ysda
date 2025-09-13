#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <unordered_map>

struct Node {
    std::unordered_map<char, Node*> children;
    size_t start;
    size_t* end;
    Node* suffix_link;  // connects a node representing a string "xα" to the
                        // node representing "α".
    size_t suffix_index;

    Node(size_t start, size_t* end)
        : start(start), end(end), suffix_link(nullptr), suffix_index(-1) {}

    size_t EdgeLen() const { return *end - start + 1; }
};

class SuffixTree {
public:
    SuffixTree()
        : size_(0),
          last_new_node_(nullptr),
          active_edge_(-1),
          active_len_(0),
          remaining_suffix_count_(0),
          leaf_end_(-1),
          root_end_(new size_t(-1)) {
        root_ = new Node(-1, root_end_);
        active_node_ = root_;
    }

    ~SuffixTree() {
        FreeNode(root_);
        delete root_end_;
    }

    void AddText(const std::string& str) {
        text_.append(str);
        for (size_t i = size_; i < text_.size(); i++) {
            ExtendSuffixTree(i);
            size_++;
        }
    }

    bool Query(const std::string& pattern) {
        Node* current = root_;
        size_t index = 0;

        while (index < pattern.size()) {
            char ch = pattern[index];

            if (current->children.find(ch) == current->children.end()) {
                return false;
            }

            Node* next = current->children[ch];
            size_t edge_len = next->EdgeLen();
            size_t index_in_edge = 0;

            while (index_in_edge < edge_len && index < pattern.size()) {
                if (text_[next->start + index_in_edge] != pattern[index]) {
                    return false;
                }
                index_in_edge++;
                index++;
            }

            if (index == pattern.size()) {
                return true;
            }

            current = next;
        }
        return true;
    }

    size_t CountNumberOfUniqueSubstrings() {
        return CountUniqueSubstrings(root_);
    }

private:
    std::string text_;
    size_t size_;

    Node* root_;
    Node* last_new_node_;

    Node* active_node_;
    size_t active_edge_;
    size_t active_len_;

    size_t remaining_suffix_count_;  // Count of suffixes yet to be added.
    size_t leaf_end_;   // Global end for all leaves (updated as text grows).
    size_t* root_end_;  // End for root (usually -1).

    void ExtendSuffixTree(size_t pos) {
        leaf_end_ = pos;
        remaining_suffix_count_++;
        last_new_node_ = nullptr;

        while (remaining_suffix_count_ > 0) {
            if (active_len_ == 0) {
                active_edge_ = pos;
            }

            char current_char = text_[active_edge_];

            if (active_node_->children.find(current_char) ==
                active_node_->children.end()) {
                CreateNewLeaf(pos, current_char);
            } else {
                Node* next = active_node_->children[current_char];
                if (active_len_ >= next->EdgeLen()) {
                    WalkDownEdge(next);
                    continue;
                }

                if (text_[next->start + active_len_] == text_[pos]) {
                    if (last_new_node_ != nullptr && active_node_ != root_) {
                        last_new_node_->suffix_link = active_node_;
                        last_new_node_ = nullptr;
                    }
                    active_len_++;
                    break;
                }
                SplitEdge(next, pos);
            }

            remaining_suffix_count_--;
            UpdateActivePoint(pos);
        }
    }

    void CreateNewLeaf(size_t pos, char current_char) {
        active_node_->children[current_char] = new Node(pos, &leaf_end_);
        if (last_new_node_ != nullptr) {
            last_new_node_->suffix_link = active_node_;
            last_new_node_ = nullptr;
        }
    }

    void WalkDownEdge(Node* next) {
        active_edge_ += next->EdgeLen();
        active_len_ -= next->EdgeLen();
        active_node_ = next;
    }

    void SplitEdge(Node* next, size_t pos) {
        size_t* split_end = new size_t;
        *split_end = next->start + active_len_ - 1;
        Node* split = new Node(next->start, split_end);

        active_node_->children[text_[active_edge_]] = split;
        split->children[text_[pos]] = new Node(pos, &leaf_end_);
        next->start += active_len_;
        split->children[text_[next->start]] = next;

        if (last_new_node_ != nullptr) {
            last_new_node_->suffix_link = split;
        }
        last_new_node_ = split;
    }

    void UpdateActivePoint(size_t pos) {
        if (active_node_ == root_ && active_len_ > 0) {
            active_len_--;
            active_edge_ = pos - remaining_suffix_count_ + 1;
        } else if (active_node_ != root_) {
            active_node_ = (active_node_->suffix_link != nullptr)
                               ? active_node_->suffix_link
                               : root_;
        }
    }

    size_t CountUniqueSubstrings(Node* node) {
        size_t count = 0;
        if (node != root_) {
            size_t edge_end = (*(node->end) < size_ ? *(node->end) : size_ - 1);
            count += edge_end - node->start + 1;
        }
        for (auto& pair : node->children) {
            count += CountUniqueSubstrings(pair.second);
        }
        return count;
    }

    void FreeNode(Node* node) {
        if (node == nullptr) {
            return;
        }
        for (auto& pair : node->children) {
            FreeNode(pair.second);
        }
        if (node->end != &leaf_end_ && node->end != root_end_) {
            delete node->end;
        }
        delete node;
    }
};

std::string ToLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char chr) { return std::tolower(chr); });
    return result;
}

int main() {
    SuffixTree st;

    std::string str;
    std::cin >> str;
    st.AddText(str);

    std::cout << st.CountNumberOfUniqueSubstrings() << std::endl;

    return 0;
}
