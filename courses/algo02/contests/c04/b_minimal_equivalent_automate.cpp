#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace std {
template <>
struct hash<std::pair<int, char>> {
    std::size_t operator()(const std::pair<int, char>& key) const {
        return std::hash<char>()(key.second) + 2U * std::hash<int>()(key.first);
    }
};
}  // namespace std

struct Automat {
    int count_states;
    int alphabet_size;
    std::unordered_map<std::pair<int, char>, std::vector<int>> inverse_graph;
    std::unordered_set<int> is_terminal;

    explicit Automat(int size, int alphabet)
        : count_states(size), alphabet_size(alphabet) {}

    void SetNewCountStates(int new_count_states) {
        count_states = new_count_states;
    }
};

struct Separator {
    int sep_class;
    char sep_symbol;
    Separator(int class_equiv, char symbol) noexcept
        : sep_class(class_equiv), sep_symbol(symbol) {}
};

class ClassDivision {
public:
    int AddNewClass() {
        classes_.emplace_back();
        return static_cast<int>(classes_.size()) - 1;
    }

    void AddElementInLastClass(int element) {
        classes_.back().insert(element);
        state_to_class_[element] = static_cast<int>(classes_.size()) - 1;
    }

    void MoveElement(int old_class, int new_class, int state) {
        classes_[old_class].erase(state);
        classes_[new_class].insert(state);
    }

    int CountEquivClass() const noexcept {
        return static_cast<int>(classes_.size());
    }

    int GetClassSize(int class_id) const {
        return static_cast<int>(classes_[class_id].size());
    }

    std::unordered_set<int>& GetClassElements(int class_id) {
        return classes_[class_id];
    }

    int GetStateClass(int state) const { return state_to_class_.at(state); }

    void SetStateClass(int state, int new_class) {
        state_to_class_[state] = new_class;
    }

    void SwapClasses(int first_class, int second_class) {
        std::swap(classes_[first_class], classes_[second_class]);
    }

private:
    std::vector<std::unordered_set<int>> classes_;
    std::unordered_map<int, int> state_to_class_;
};

static std::unordered_map<int, std::vector<int>> ConstructInvolvedState(
    ClassDivision& division, Automat& automaton, const Separator& sep) {
    std::unordered_map<int, std::vector<int>> involved;

    auto& inv = automaton.inverse_graph;

    for (int state : division.GetClassElements(sep.sep_class)) {
        auto it = inv.find({state, sep.sep_symbol});

        if (it == inv.end()) {
            continue;
        }

        for (int predecessor : it->second) {
            int pred_class = division.GetStateClass(predecessor);

            involved[pred_class].push_back(predecessor);
        }
    }

    return involved;
}

static void SeparatingAllStates(
    ClassDivision& division,
    std::unordered_map<int, std::vector<int>>& involved,

    std::queue<Separator>& work_queue, Automat& automaton) {
    for (auto& [class_id, states] : involved) {
        if (static_cast<int>(states.size()) < division.GetClassSize(class_id)) {
            int new_class = division.AddNewClass();

            for (int st : states) {
                division.MoveElement(class_id, new_class, st);
            }

            if (division.GetClassSize(class_id) <
                division.GetClassSize(new_class)) {
                division.SwapClasses(class_id, new_class);
            }

            for (int st : division.GetClassElements(new_class)) {
                division.SetStateClass(st, new_class);
            }

            for (int sym = 0; sym < automaton.alphabet_size; ++sym) {
                work_queue.emplace(new_class, char('a' + sym));
            }
        }
    }
}

static int FindCountClassEquivalence(
    Automat& automaton, const std::unordered_set<int>& reachable_states) {
    ClassDivision division;

    if (!automaton.is_terminal.empty()) {
        division.AddNewClass();
        for (int t_state : automaton.is_terminal) {
            division.AddElementInLastClass(t_state);
        }
    }

    if (automaton.count_states -
            static_cast<int>(automaton.is_terminal.size()) >
        0) {
        division.AddNewClass();

        for (int st : reachable_states) {
            if (!automaton.is_terminal.contains(st)) {
                division.AddElementInLastClass(st);
            }
        }
    }

    std::queue<Separator> work_queue;
    int initial_classes = division.CountEquivClass();

    for (int cls = 0; cls < initial_classes; ++cls) {
        for (int sym = 0; sym < automaton.alphabet_size; ++sym) {
            work_queue.emplace(cls, char('a' + sym));
        }
    }

    while (!work_queue.empty()) {
        Separator sep = work_queue.front();
        work_queue.pop();

        auto involved = ConstructInvolvedState(division, automaton, sep);

        SeparatingAllStates(division, involved, work_queue, automaton);
    }

    return division.CountEquivClass();
}

static std::unordered_set<int> BFS(
    const std::vector<std::unordered_map<char, int>>& graph) {
    std::queue<int> que;
    que.push(0);

    std::unordered_set<int> visited;
    visited.insert(0);

    while (!que.empty()) {
        int curr = que.front();
        que.pop();
        for (const auto& [sym, nxt] : graph[curr]) {
            if (!visited.contains(nxt)) {
                visited.insert(nxt);

                que.push(nxt);
            }
        }
    }

    return visited;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int count_states;
    int count_terminal_states;
    int alphabet_size;
    std::cin >> count_states >> count_terminal_states >> alphabet_size;

    std::vector<std::unordered_map<char, int>> graph(count_states);
    std::unordered_set<int> terminals;
    terminals.reserve(count_terminal_states);

    for (int i = 0; i < count_terminal_states; ++i) {
        int term_state;
        std::cin >> term_state;
        terminals.insert(term_state);
    }

    for (int i = 0; i < count_states * alphabet_size; ++i) {
        int src;
        char sym;
        int dst;
        std::cin >> src >> sym >> dst;
        graph[src][sym] = dst;
    }

    auto reachable = BFS(graph);
    Automat automaton(static_cast<int>(reachable.size()), alphabet_size);

    for (int ter : terminals) {
        if (reachable.contains(ter)) {
            automaton.is_terminal.insert(ter);
        }
    }

    for (int src = 0; src < count_states; ++src) {
        if (!reachable.contains(src)) {
            continue;
        }
        for (auto& [sym, dst] : graph[src]) {
            automaton.inverse_graph[{dst, sym}].push_back(src);
        }
    }

    std::cout << FindCountClassEquivalence(automaton, reachable) << '\n';

    return 0;
}
