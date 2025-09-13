#include <algorithm>
#include <cstddef>
#include <cstring>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <class Iterator>
class IteratorRange {
public:
    IteratorRange(Iterator begin, Iterator end) : begin_(begin), end_(end) {}

    Iterator begin() const { return begin_; }  // NOLINT
    Iterator end() const { return end_; }      // NOLINT

private:
    Iterator begin_, end_;
};

namespace traverses {  // NOLINT

// Traverses the connected component in a breadth-first order
// from the vertex 'origin_vertex'.
// Refer to
// https://goo.gl/0qYXzC
// for the visitor events.
template <class Vertex, class Graph, class Visitor>
void BreadthFirstSearch(Vertex origin_vertex, const Graph& graph,
                        Visitor visitor) {
    std::queue<Vertex> queue;
    std::unordered_set<Vertex> visited;

    queue.push(origin_vertex);
    visited.insert(origin_vertex);

    visitor.DiscoverVertex(origin_vertex);

    while (!queue.empty()) {
        Vertex current = queue.front();
        queue.pop();

        visitor.ExamineVertex(current);

        for (auto& edge : OutgoingEdges(graph, current)) {
            visitor.ExamineEdge(edge);

            Vertex next = GetTarget(graph, edge);

            auto [it, inserted] = visited.insert(next);
            if (inserted) {
                visitor.DiscoverVertex(next);
                queue.push(next);
            }
        }
    }
}

// See "Visitor Event Points" on
// https://goo.gl/wtAl0y
template <class Vertex, class Edge>
class BfsVisitor {
public:
    virtual void DiscoverVertex(Vertex /*vertex*/) {}
    virtual void ExamineEdge(const Edge& /*edge*/) {}
    virtual void ExamineVertex(Vertex /*vertex*/) {}
    virtual ~BfsVisitor() = default;
};

}  // namespace traverses

namespace aho_corasick {  // NOLINT

struct AutomatonNode {
    AutomatonNode() : suffix_link(nullptr), terminal_link(nullptr) {}

    // Stores ids of strings which are ended at this node.
    std::vector<size_t> terminated_string_ids;
    // Stores tree structure of nodes.
    std::map<char, AutomatonNode> trie_transitions;
    // Stores cached transitions of the automaton, contains
    // only pointers to the elements of trie_transitions.
    std::map<char, AutomatonNode*> cache;
    AutomatonNode* suffix_link;
    AutomatonNode* terminal_link;
};

inline AutomatonNode* GetTrieTransition(AutomatonNode* node, char character) {
    auto it = node->trie_transitions.find(character);

    if (it == node->trie_transitions.end()) {
        return nullptr;
    }

    return &it->second;
}

// Provides constant amortized runtime.
inline AutomatonNode* GetAutomatonTransition(AutomatonNode* node,
                                             const AutomatonNode* root,
                                             char character) {
    if (node->cache.find(character) != node->cache.end()) {
        return node->cache[character];
    }

    AutomatonNode* transition = GetTrieTransition(node, character);

    if (transition == nullptr) {
        if (node == root) {
            transition = node;
        } else {
            transition =
                GetAutomatonTransition(node->suffix_link, root, character);
        }
    }

    node->cache[character] = transition;

    return node->cache[character];
}

namespace internal {  // NOLINT

class AutomatonGraph {
public:
    struct Edge {
        Edge(AutomatonNode* source, AutomatonNode* target, char character)
            : source(source), target(target), character(character) {}

        AutomatonNode* source;
        AutomatonNode* target;
        char character;
    };
};

inline std::vector<AutomatonGraph::Edge> OutgoingEdges(
    const AutomatonGraph& /*graph*/, AutomatonNode* vertex) {
    std::vector<AutomatonGraph::Edge> edges;
    edges.reserve(vertex->trie_transitions.size());

    for (auto& [character, child] : vertex->trie_transitions) {
        edges.emplace_back(vertex, &child, character);
    }

    return edges;
}

inline AutomatonNode* GetTarget(const AutomatonGraph& /*graph*/,
                                const AutomatonGraph::Edge& edge) {
    return edge.target;
}

class SuffixLinkCalculator
    : public traverses::BfsVisitor<AutomatonNode*, AutomatonGraph::Edge> {
public:
    explicit SuffixLinkCalculator(AutomatonNode* root) : root_(root) {}

    void ExamineEdge(const AutomatonGraph::Edge& edge) override {
        if (edge.source == root_) {
            edge.target->suffix_link = root_;
            return;
        }

        edge.target->suffix_link = GetAutomatonTransition(
            edge.source->suffix_link, root_, edge.character);
    }

private:
    AutomatonNode* root_;
};

class TerminalLinkCalculator
    : public traverses::BfsVisitor<AutomatonNode*, AutomatonGraph::Edge> {
public:
    explicit TerminalLinkCalculator(AutomatonNode* root) : root_(root) {}

    void DiscoverVertex(AutomatonNode* vertex) override {
        if (vertex == root_) {
            return;
        }

        AutomatonNode* suffix = vertex->suffix_link;

        if (!suffix->terminated_string_ids.empty()) {
            vertex->terminal_link = suffix;
        } else {
            if (suffix != nullptr) {
                vertex->terminal_link = suffix->terminal_link;
            } else {
                vertex->terminal_link = nullptr;
            }
        }
    }

private:
    AutomatonNode* root_;
};

}  // namespace internal

class NodeReference {
public:
    NodeReference() : node_(nullptr), root_(nullptr) {}

    NodeReference(AutomatonNode* node, AutomatonNode* root)
        : node_(node), root_(root) {}

    NodeReference Next(char character) const {
        if (node_ == nullptr) {
            return NodeReference();
        }
        AutomatonNode* next_node =
            GetAutomatonTransition(node_, root_, character);
        return NodeReference(next_node, root_);
    }

    template <class Callback>
    void GenerateMatches(Callback on_match) const {
        if (node_ == nullptr) {
            return;
        }

        for (size_t id : TerminatedStringIds()) {
            on_match(id);
        }

        NodeReference term = TerminalLink();
        if (term) {
            term.GenerateMatches(on_match);
        }
    }

    bool IsTerminal() const {
        return (node_ != nullptr) && !node_->terminated_string_ids.empty();
    }

    explicit operator bool() const { return node_ != nullptr; }

    bool operator==(NodeReference other) const { return node_ == other.node_; }

private:
    using TerminatedStringIterator = std::vector<size_t>::const_iterator;
    using TerminatedStringIteratorRange =
        IteratorRange<TerminatedStringIterator>;

    NodeReference TerminalLink() const {
        if ((node_ != nullptr) && (node_->terminal_link != nullptr)) {
            return NodeReference(node_->terminal_link, root_);
        }

        return NodeReference();
    }

    TerminatedStringIteratorRange TerminatedStringIds() const {
        return {node_->terminated_string_ids.begin(),
                node_->terminated_string_ids.end()};
    }

    AutomatonNode* node_;
    AutomatonNode* root_;
};

class AutomatonBuilder;

class Automaton {
public:
    Automaton(const Automaton&) = delete;
    Automaton& operator=(const Automaton&) = delete;

    NodeReference Root() { return NodeReference(&root_, &root_); }

private:
    Automaton() = default;

    AutomatonNode root_;

    friend class AutomatonBuilder;
};

class AutomatonBuilder {
public:
    void Add(const std::string& string, size_t id) {
        words_.push_back(string);
        ids_.push_back(id);
    }

    std::unique_ptr<Automaton> Build() {
        std::unique_ptr<Automaton> automaton(new Automaton());

        BuildTrie(words_, ids_, automaton.get());
        BuildSuffixLinks(automaton.get());
        BuildTerminalLinks(automaton.get());

        return automaton;
    }

private:
    static void BuildTrie(const std::vector<std::string>& words,
                          const std::vector<size_t>& ids,
                          Automaton* automaton) {
        for (size_t i = 0; i < words.size(); ++i) {
            AddString(&automaton->root_, ids[i], words[i]);
        }
    }

    static void AddString(AutomatonNode* root, size_t string_id,
                          const std::string& string) {
        AutomatonNode* current = root;

        for (char character : string) {
            current = &current->trie_transitions[character];
        }

        current->terminated_string_ids.push_back(string_id);
    }

    static void BuildSuffixLinks(Automaton* automaton) {
        using namespace internal;

        AutomatonNode* root = &automaton->root_;
        root->suffix_link = root;

        AutomatonGraph graph;
        SuffixLinkCalculator visitor(root);

        traverses::BreadthFirstSearch(root, graph, visitor);
    }

    static void BuildTerminalLinks(Automaton* automaton) {
        using namespace internal;

        AutomatonNode* root = &automaton->root_;

        AutomatonGraph graph;
        TerminalLinkCalculator visitor(root);

        traverses::BreadthFirstSearch<AutomatonNode*, AutomatonGraph,
                                      TerminalLinkCalculator>(root, graph,
                                                              visitor);
    }

    std::vector<std::string> words_;
    std::vector<size_t> ids_;
};

}  // namespace aho_corasick

template <class Predicate>
std::vector<std::string> Split(const std::string& string,
                               Predicate is_delimiter) {
    std::vector<std::string> patterns;

    size_t left = 0;
    for (size_t right = 0; right < string.size(); ++right) {
        if (is_delimiter(string[right])) {
            patterns.push_back(string.substr(left, right - left));

            left = right + 1;
        }
    }

    patterns.push_back(string.substr(left));

    return patterns;
}

class WildcardMatcher {
public:
    WildcardMatcher() : number_of_words_(0), pattern_len_(0), index_(0) {}

    WildcardMatcher static BuildFor(const std::string& pattern, char wildcard) {
        WildcardMatcher new_matcher;
        new_matcher.pattern_len_ = pattern.size();

        std::vector<std::string> segments =
            Split(pattern, [wildcard](char ch) { return wildcard == ch; });

        size_t len = 0;
        aho_corasick::AutomatonBuilder builder;

        for (auto& part : segments) {
            if (part.empty()) {
                ++len;
                continue;
            }

            new_matcher.words_offsets_.push_back(len);
            len += part.size();
            builder.Add(part, new_matcher.pattern_len_ - len);

            ++len;
        }

        new_matcher.aho_corasick_automaton_ = builder.Build();
        new_matcher.number_of_words_ = new_matcher.words_offsets_.size();

        new_matcher.Reset();

        return new_matcher;
    }

    void Reset() {
        state_ = aho_corasick_automaton_->Root();
        words_occurrences_by_position_.assign(pattern_len_, 0);
        index_ = 0;
    }

    template <class Callback>
    void Scan(char character, Callback on_match) {
        state_ = state_.Next(character);
        ++index_;

        state_.GenerateMatches([this](size_t shift) {
            int match_position = shift;

            if (shift >= 0 && shift < static_cast<int>(pattern_len_)) {
                ++words_occurrences_by_position_[shift];
            }
        });

        if (index_ >= pattern_len_ &&
            words_occurrences_by_position_[0] == number_of_words_) {
            on_match();
        }

        words_occurrences_by_position_.pop_front();
        words_occurrences_by_position_.push_back(0);
    }

private:
    std::deque<size_t> words_occurrences_by_position_;
    std::vector<size_t> words_offsets_;
    size_t number_of_words_;
    size_t pattern_len_;
    size_t index_;
    aho_corasick::NodeReference state_;
    std::unique_ptr<aho_corasick::Automaton> aho_corasick_automaton_;
};

std::string ReadString(std::istream& input_stream) {
    std::string str;
    std::getline(input_stream, str);

    return str;
}

std::vector<size_t> FindFuzzyMatches(const std::string& pattern_with_wildcards,
                                     const std::string& text, char wildcard) {
    WildcardMatcher matcher =
        WildcardMatcher::BuildFor(pattern_with_wildcards, wildcard);

    std::vector<size_t> matches;

    size_t pattern_len = pattern_with_wildcards.size() - 1;

    for (size_t i = 0; i < text.size(); ++i) {
        matcher.Scan(text[i], [&matches, i, pattern_len]() {
            matches.push_back(i - pattern_len);
        });
    }

    return matches;
}

void Print(const std::vector<size_t>& sequence) {
    std::cout << sequence.size() << '\n';

    std::copy(sequence.begin(), sequence.end(),
              std::ostream_iterator<size_t>(std::cout, " "));

    std::cout << '\n';
}
