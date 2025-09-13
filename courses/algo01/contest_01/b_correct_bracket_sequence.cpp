#include <iostream>
#include <stack>
#include <string>

bool is_matching_pair(char open_bracket, char close_bracket) {
    return (open_bracket == '(' && close_bracket == ')') ||
           (open_bracket == '[' && close_bracket == ']') ||
           (open_bracket == '{' && close_bracket == '}');
}

void brackets(const std::string& s) {
    std::stack<char> open_brackets;

    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '(' || s[i] == '[' || s[i] == '{') {
            open_brackets.push(s[i]);
        } else {
            if (!open_brackets.empty() && is_matching_pair(open_brackets.top(), s[i])) {
                open_brackets.pop();
            } else {
                std::cout << i << '\n';
                return;
            }
        }
    }

    if (open_brackets.empty()) {
        std::cout << "CORRECT" << '\n';
    } else {
        std::cout << s.size() << '\n';
    }

    return;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string s;
    std::getline(std::cin, s);

    brackets(s);

    return 0;
}
