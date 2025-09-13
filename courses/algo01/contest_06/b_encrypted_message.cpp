#include <bits/stdc++.h>

class Treap {
private:
    struct Node {
        char ch;
        int priority;
        int subtree_size;
        Node* left;
        Node* right;
        Node(char character)
            : ch(character),
              priority(std::rand()),
              subtree_size(1),
              left(nullptr),
              right(nullptr) {}
    };

    Node* root_ = nullptr;

    static int GetSize(Node* current) {
        return (current != nullptr) ? current->subtree_size : 0;
    }

    static void UpdateSize(Node* current) {
        if (current == nullptr) {
            return;
        }
        current->subtree_size =
            GetSize(current->left) + GetSize(current->right) + 1;
    }

    void Split(Node* current, int key, Node*& left_part, Node*& right_part) {
        if (current == nullptr) {
            left_part = nullptr;
            right_part = nullptr;
            return;
        }

        int current_index = GetSize(current->left) + 1;

        if (current_index <= key) {
            Split(current->right, key - current_index, current->right,
                  right_part);
            left_part = current;
            UpdateSize(left_part);
        } else {
            Split(current->left, key, left_part, current->left);
            right_part = current;
            UpdateSize(right_part);
        }
    }

    Node* Merge(Node* left_part, Node* right_part) {
        if (left_part == nullptr) {
            return right_part;
        }
        if (right_part == nullptr) {
            return left_part;
        }
        if (left_part->priority > right_part->priority) {
            left_part->right = Merge(left_part->right, right_part);
            UpdateSize(left_part);

            return left_part;
        }

        right_part->left = Merge(left_part, right_part->left);
        UpdateSize(right_part);

        return right_part;
    }

    void InorderTraversal(Node* current, std::string& result) const {
        if (current == nullptr) {
            return;
        }

        InorderTraversal(current->left, result);

        result.push_back(current->ch);

        InorderTraversal(current->right, result);
    }

    void Clear(Node* current) {
        if (current == nullptr) {
            return;
        }

        Clear(current->left);
        Clear(current->right);

        delete current;
    }

public:
    ~Treap() { Clear(root_); }

    void BuildFromString(const std::string& input_string) {
        for (auto character : input_string) {
            Node* new_node = new Node(character);
            root_ = Merge(root_, new_node);
        }
    }

    void LeftRotateSubstring(int start, int end, int shift_count) {
        Node* left_part;
        Node* subtree_to_split;
        Node* extracted_segment;
        Node* remaining_segment;

        Split(root_, start - 1, left_part, subtree_to_split);

        int segment_length = end - start + 1;
        Split(subtree_to_split, segment_length, extracted_segment,
              remaining_segment);

        Node* c_left;
        Node* c_right;
        Split(extracted_segment, shift_count, c_left, c_right);

        extracted_segment = Merge(c_right, c_left);

        root_ = Merge(left_part, Merge(extracted_segment, remaining_segment));
    }

    std::string GetString() const {
        std::string result;
        result.reserve(static_cast<std::size_t>(GetSize(root_)));

        InorderTraversal(root_, result);

        return result;
    }
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string encrypted_message;
    std::cin >> encrypted_message;

    int shifts_count;
    std::cin >> shifts_count;
    std::vector<std::array<int, 3>> shifts(shifts_count);
    for (int index = 0; index < shifts_count; index++) {
        std::cin >> shifts[index][0] >> shifts[index][1] >> shifts[index][2];
    }

    std::srand((unsigned)std::time(NULL));
    Treap treap_object;
    treap_object.BuildFromString(encrypted_message);

    for (int index = shifts_count - 1; index >= 0; index--) {
        int start = shifts[index][0];
        int end = shifts[index][1];
        int shift_count = shifts[index][2];

        treap_object.LeftRotateSubstring(start, end, shift_count);
    }

    std::cout << treap_object.GetString() << "\n";
    return 0;
}
