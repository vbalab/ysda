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
inline void BreadthFirstSearch(Vertex origin_vertex, const Graph& graph,
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

            if (visited.find(next) == visited.end()) {
                visited.insert(next);

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
    std::map<char, AutomatonNode*> automaton_transitions_cache;
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
    if (node->automaton_transitions_cache.count(character) == 0) {
        AutomatonNode* transition = GetTrieTransition(node, character);

        if (transition == nullptr) {
            if (node == root) {
                transition = node;
            } else {
                transition =
                    GetAutomatonTransition(node->suffix_link, root, character);
            }
        }

        node->automaton_transitions_cache[character] = transition;
    }

    return node->automaton_transitions_cache[character];
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

inline std::vector<typename AutomatonGraph::Edge> OutgoingEdges(
    const AutomatonGraph& /*graph*/, AutomatonNode* vertex) {
    std::vector<AutomatonGraph::Edge> edges;

    for (auto& [chr, child] : vertex->trie_transitions) {
        edges.emplace_back(vertex, &child, chr);
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

    inline void ExamineEdge(const AutomatonGraph::Edge& edge) override {
        AutomatonNode* source = edge.source;
        AutomatonNode* child = edge.target;
        char ch = edge.character;

        if (source == root_) {
            child->suffix_link = root_;
            return;
        }

        AutomatonNode* fallback = source->suffix_link;
        while (fallback != root_ && fallback->trie_transitions.find(ch) ==
                                        fallback->trie_transitions.end()) {
            fallback = fallback->suffix_link;
        }

        if (fallback->trie_transitions.find(ch) !=
            fallback->trie_transitions.end()) {
            child->suffix_link = &fallback->trie_transitions[ch];
        } else {
            child->suffix_link = root_;
        }
    }

private:
    AutomatonNode* root_;
};

class TerminalLinkCalculator
    : public traverses::BfsVisitor<AutomatonNode*, AutomatonGraph::Edge> {
public:
    explicit TerminalLinkCalculator(AutomatonNode* root) : root_(root) {}

    inline void DiscoverVertex(AutomatonNode* vertex) override {
        if (vertex == root_) {
            return;
        }

        AutomatonNode* suffix = vertex->suffix_link;

        if ((suffix != nullptr) && !suffix->terminated_string_ids.empty()) {
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

    inline NodeReference Next(char character) const {
        if (node_ == nullptr) {
            return NodeReference();
        }
        AutomatonNode* next_node =
            GetAutomatonTransition(node_, root_, character);
        return NodeReference(next_node, root_);
    }

    template <class Callback>
    inline void GenerateMatches(Callback on_match) const {
        if (node_ == nullptr) {
            return;
        }

        for (auto it = node_->terminated_string_ids.begin();
             it != node_->terminated_string_ids.end(); ++it) {
            on_match(*it);
        }

        NodeReference term = TerminalLink();
        if (term) {
            term.GenerateMatches(on_match);
        }
    }

    inline bool IsTerminal() const {
        return (node_ != nullptr) && !node_->terminated_string_ids.empty();
    }

    explicit operator bool() const { return node_ != nullptr; }

    inline bool operator==(NodeReference other) const {
        return node_ == other.node_;
    }

private:
    using TerminatedStringIterator = std::vector<size_t>::const_iterator;
    using TerminatedStringIteratorRange =
        IteratorRange<TerminatedStringIterator>;

    inline NodeReference TerminalLink() const {
        if ((node_ != nullptr) && (node_->terminal_link != nullptr)) {
            return NodeReference(node_->terminal_link, root_);
        }

        return NodeReference();
    }

    inline TerminatedStringIteratorRange TerminatedStringIds() const {
        return TerminatedStringIteratorRange(
            node_->terminated_string_ids.begin(),
            node_->terminated_string_ids.end());
    }

    AutomatonNode* node_;
    AutomatonNode* root_;
};

class AutomatonBuilder;

class Automaton {
public:
    Automaton() = default;

    Automaton(const Automaton&) = delete;
    Automaton& operator=(const Automaton&) = delete;

    inline NodeReference Root() { return NodeReference(&root_, &root_); }

private:
    AutomatonNode root_;

    friend class AutomatonBuilder;
};

class AutomatonBuilder {
public:
    inline void Add(const std::string& string, size_t id) {
        words_.push_back(string);
        ids_.push_back(id);
    }

    std::unique_ptr<Automaton> Build() {
        std::unique_ptr<Automaton> automaton = std::make_unique<Automaton>();

        BuildTrie(words_, ids_, automaton.get());
        BuildSuffixLinks(automaton.get());
        BuildTerminalLinks(automaton.get());

        return automaton;
    }

private:
    inline static void BuildTrie(const std::vector<std::string>& words,
                                 const std::vector<size_t>& ids,
                                 Automaton* automaton) {
        for (size_t i = 0; i < words.size(); ++i) {
            AddString(&automaton->root_, ids[i], words[i]);
        }
    }

    inline static void AddString(AutomatonNode* root, size_t string_id,
                                 const std::string& string) {
        AutomatonNode* cur = root;

        for (char ch : string) {
            if (cur->trie_transitions.count(ch) == 0) {
                cur->trie_transitions[ch] = AutomatonNode();
            }

            cur = &cur->trie_transitions[ch];
        }

        cur->terminated_string_ids.push_back(string_id);
    }

    inline static void BuildSuffixLinks(Automaton* automaton) {
        using namespace internal;

        AutomatonNode* root = &automaton->root_;
        root->suffix_link = root;

        AutomatonGraph graph;
        SuffixLinkCalculator visitor(root);

        traverses::BreadthFirstSearch<AutomatonNode*, AutomatonGraph,
                                      SuffixLinkCalculator>(root, graph,
                                                            visitor);
    }

    inline static void BuildTerminalLinks(Automaton* automaton) {
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
inline std::vector<std::string> Split(const std::string& string,
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

    static WildcardMatcher BuildFor(const std::string& pattern, char wildcard) {
        WildcardMatcher matcher;

        matcher.pattern_len_ = pattern.size();

        std::vector<std::string> segments =
            Split(pattern, [wildcard](char ch) { return ch == wildcard; });

        size_t pos = 0;
        for (auto& segment : segments) {
            while (pos < pattern.size() && pattern[pos] == wildcard) {
                ++pos;
            }
            matcher.segments_.emplace_back(pos, segment.size());
            pos += segment.size();
        }
        matcher.number_of_words_ = matcher.segments_.size();

        aho_corasick::AutomatonBuilder builder;
        for (size_t i = 0; i < segments.size(); ++i) {
            builder.Add(segments[i], i);
        }
        matcher.aho_corasick_automaton_ = builder.Build();

        matcher.Reset();

        return matcher;
    }

    inline void Reset() {
        vote_counts_.clear();
        index_ = 0;

        state_ = aho_corasick_automaton_->Root();
    }

    template <class Callback>
    inline void Scan(char character, Callback on_match) {
        state_ = state_.Next(character);

        state_.GenerateMatches([this](size_t segment_id) {
            size_t seg_offset = segments_[segment_id].first;
            size_t seg_length = segments_[segment_id].second;

            size_t candidate =
                static_cast<size_t>(index_) - (seg_offset + seg_length - 1);

            vote_counts_[candidate]++;
        });

        if (index_ + 1 >= pattern_len_) {
            size_t candidate = index_ + 1 - pattern_len_;

            if (vote_counts_[candidate] == number_of_words_) {
                on_match();
            }
        }

        ++index_;
    }

private:
    size_t number_of_words_;
    size_t pattern_len_;
    size_t index_;
    std::unordered_map<size_t, size_t> vote_counts_;
    std::vector<std::pair<size_t, size_t>> segments_;
    aho_corasick::NodeReference state_;
    std::unique_ptr<aho_corasick::Automaton> aho_corasick_automaton_;
};

inline std::string ReadString(std::istream& input_stream) {
    std::string str;
    std::getline(input_stream, str);

    return str;
}

inline std::vector<size_t> FindFuzzyMatches(
    const std::string& pattern_with_wildcards, const std::string& text,
    char wildcard) {
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

inline void Print(const std::vector<size_t>& sequence) {
    std::cout << sequence.size() << '\n';

    std::copy(sequence.begin(), sequence.end(),
              std::ostream_iterator<size_t>(std::cout, " "));

    std::cout << '\n';
}
