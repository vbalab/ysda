/*
 * Интерфейс прокомментирован с целью объяснить,
 * почему он написан так, а не иначе. В реальной жизни
 * так никто никогда не делает. Комментарии к коду,
 * которые остались бы в его рабочем варианте, заданы
 * с помощью команды однострочного комментария // и написаны
 * на английском языке, как рекомендуется.
 * Остальные комментарии здесь исключительно в учебных целях.
 */

#include <algorithm>
#include <cstring>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>


/*
 * Часто в c++ приходится иметь дело с парой итераторов,
 * которые представляют из себя полуинтервал. Например,
 * функция std:sort принимает пару итераторов, участок
 * между которыми нужно отсортировать. В с++11 появился
 * удобный range-based for, который позволяет итерироваться
 * по объекту, для которого определены функции std::begin
 * и std::end (например, это объекты: массив фиксированного
 * размера, любой объект, у которого определены методы
 * begin() и end()). То есть удобный способ итерироваться
 * по std::vector такой:
 * for (const std::string& string: words).
 * Однако, для некоторых объектов существует не один способ
 * итерироваться по ним. Например std::map: мы можем
 * итерироваться по парам объект-значение (как это сделает
 * for(...: map)), или мы можем итерироваться по ключам.
 * Для этого мы можем сделать функцию:
 * IteratorRange<...> keys(const std::map& map),
 * которой можно удобно воспользоваться:
 * for(const std::string& key: keys(dictionary)).
 */
template <class Iterator>
class IteratorRange {
public:
    IteratorRange(Iterator begin, Iterator end) : begin_(begin), end_(end) {}

    Iterator begin() const { return begin_; }
    Iterator end() const { return end_; }

private:
    Iterator begin_, end_;
};

namespace traverses {

// Traverses the connected component in a breadth-first order
// from the vertex 'origin_vertex'.
// Refer to
// https://goo.gl/0qYXzC
// for the visitor events.
template <class Vertex, class Graph, class Visitor>
void BreadthFirstSearch(Vertex origin_vertex, const Graph &graph,
                        Visitor visitor);

/*
 * Для начала мы рекомендуем ознакомиться с общей
 * концепцией паттерна проектирования Visitor:
 * https://goo.gl/oZGiYl
 * Для применения Visitor'а к задаче обхода графа
 * можно ознакомиться с
 * https://goo.gl/5gjef2
 */
// See "Visitor Event Points" on
// https://goo.gl/wtAl0y
template <class Vertex, class Edge>
class BfsVisitor {
public:
    virtual void DiscoverVertex(Vertex /*vertex*/) {}
    virtual void ExamineEdge(const Edge & /*edge*/) {}
    virtual void ExamineVertex(Vertex /*vertex*/) {}
    virtual ~BfsVisitor() = default;
};

} // namespace traverses

namespace aho_corasick {

struct AutomatonNode {
    AutomatonNode() : suffix_link(nullptr), terminal_link(nullptr) {}

    // Stores ids of strings which are ended at this node.
    std::vector<size_t> terminated_string_ids;
    // Stores tree structure of nodes.
    std::map<char, AutomatonNode> trie_transitions;
    /*
     * Обратите внимание, что std::set/std::map/std::list
     * при вставке и удалении неинвалидируют ссылки на
     * остальные элементы контейнера. Но стандартные контейнеры
     * std::vector/std::string/std::deque таких гарантий не
     * дают, поэтому хранение указателей на элементы этих
     * контейнеров крайне не рекомендуется.
     */
    // Stores cached transitions of the automaton, contains
    // only pointers to the elements of trie_transitions.
    std::map<char, AutomatonNode *> automaton_transitions_cache;
    AutomatonNode *suffix_link;
    AutomatonNode *terminal_link;
};

// Returns a corresponding trie transition 'nullptr' otherwise.
AutomatonNode *GetTrieTransition(AutomatonNode *node, char character);

// Returns an automaton transition, updates 'node->automaton_transitions_cache'
// if necessary.
// Provides constant amortized runtime.
AutomatonNode *GetAutomatonTransition(AutomatonNode *node,
                                      const AutomatonNode *root,
                                      char character);

namespace internal {

class AutomatonGraph {
public:
    struct Edge {
        Edge(AutomatonNode *source, AutomatonNode *target, char character)
            : source(source), target(target), character(character) {}

        AutomatonNode *source;
        AutomatonNode *target;
        char character;
    };
};

std::vector<typename AutomatonGraph::Edge> OutgoingEdges(
    const AutomatonGraph & /*graph*/, AutomatonNode *vertex);

AutomatonNode *GetTarget(const AutomatonGraph & /*graph*/,
                         const AutomatonGraph::Edge &edge);

class SuffixLinkCalculator
    : public traverses::BfsVisitor<AutomatonNode *, AutomatonGraph::Edge> {
public:
    explicit SuffixLinkCalculator(AutomatonNode *root) : root_(root) {}

    void ExamineVertex(AutomatonNode *node) override;

    void ExamineEdge(const AutomatonGraph::Edge &edge) override;

private:
    AutomatonNode *root_;
};

class TerminalLinkCalculator
    : public traverses::BfsVisitor<AutomatonNode *, AutomatonGraph::Edge> {
public:
    explicit TerminalLinkCalculator(AutomatonNode *root) : root_(root) {}

    /*
     * Если вы не знакомы с ключевым словом override,
     * то ознакомьтесь
     * https://goo.gl/u024X0
     */
    void DiscoverVertex(AutomatonNode *node) override;

private:
    AutomatonNode *root_;
};

} // namespace internal

/*
 * Объясним задачу, которую решает класс NodeReference.
 * Класс Automaton представляет из себя неизменяемый объект
 * (https://goo.gl/4rSP4f),
 * в данном случае, это означает, что единственное действие,
 * которое пользователь может совершать с готовым автоматом,
 * это обходить его разными способами. Это значит, что мы
 * должны предоставить пользователю способ получить вершину
 * автомата и дать возможность переходить между вершинами.
 * Одним из способов это сделать -- это предоставить
 * пользователю константный указатель на AutomatonNode,
 * а вместе с ним константый интерфейс AutomatonNode. Однако,
 * этот вариант ведет к некоторым проблемам.
 * Во-первых, этот же интерфейс AutomatonNode мы должны
 * использовать и для общения автомата с этим внутренним
 * представлением вершины. Так как константная версия
 * этого же интерфейса будет доступна пользователю, то мы
 * ограничены в добавлении функций в этот константный
 * интерфейс (не все функции, которые должны быть доступны
 * автомату должны быть доступны пользователю). Во-вторых,
 * так как мы используем кэширование при переходе по символу
 * в автомате, то условная функция getNextNode не может быть
 * константной (она заполняет кэш переходов). Это значит,
 * что мы лишены возможности добавить функцию "перехода
 * между вершинами" в константный интерфейс (то есть,
 * предоставить ее пользователю константного указателя на
 * AutomatonNode).
 * Во избежание этих проблем, мы создаем отдельный
 * класс, отвечающий ссылке на вершину, который предоставляет
 * пользователю только нужный интерфейс.
 */
class NodeReference {
public:
    NodeReference() : node_(nullptr), root_(nullptr) {}

    NodeReference(AutomatonNode *node, AutomatonNode *root)
        : node_(node), root_(root) {}

    NodeReference Next(char character) const;

    /*
     * В этом случае есть два хороших способа получить
     * результат работы этой функции:
     * добавить параметр типа OutputIterator, который
     * последовательно записывает в него id найденных
     * строк, или же добавить параметр типа Callback,
     * который будет вызван для каждого такого id.
     * Чтобы ознакомиться с этими концепциями лучше,
     * смотрите ссылки:
     * https://goo.gl/2Kg8wE
     * https://goo.gl/OaUB4k
     * По своей мощности эти способы эквивалентны. (см.
     * https://goo.gl/UaQpPq)
     * Так как в интерфейсе WildcardMatcher мы принимаем
     * Callback, то чтобы пользоваться одним и тем же средством
     * во всех интерфейсах, мы и здесь применяем Callback. Отметим,
     * что другие способы, как например, вернуть std::vector с
     * найденными id, не соответствуют той же степени гибкости, как
     * 2 предыдущие решения (чтобы в этом убедиться представьте
     * себе, как можно решить такую задачу: создать std::set
     * из найденных id).
     */
    template <class Callback>
    void GenerateMatches(Callback on_match) const;

    bool IsTerminal() const;

    explicit operator bool() const { return node_ != nullptr; }

    bool operator==(NodeReference other) const;

private:
    using TerminatedStringIterator = std::vector<size_t>::const_iterator;
    using TerminatedStringIteratorRange = IteratorRange<TerminatedStringIterator>;

    NodeReference TerminalLink() const;

    TerminatedStringIteratorRange TerminatedStringIds() const;

    AutomatonNode *node_;
    AutomatonNode *root_;
};

class AutomatonBuilder;

class Automaton {
public:
    /*
     * Чтобы ознакомиться с конструкцией =default, смотрите
     * https://goo.gl/jixjHU
     */
    Automaton() = default;

    Automaton(const Automaton &) = delete;
    Automaton &operator=(const Automaton &) = delete;

    NodeReference Root();

private:
    AutomatonNode root_;

    friend class AutomatonBuilder;
};

class AutomatonBuilder {
public:
    void Add(const std::string &string, size_t id);

    std::unique_ptr<Automaton> Build() {
        auto automaton = std::make_unique<Automaton>();
        BuildTrie(words_, ids_, automaton.get());
        BuildSuffixLinks(automaton.get());
        BuildTerminalLinks(automaton.get());
        return automaton;
    }

private:
    static void BuildTrie(const std::vector<std::string> &words,
                          const std::vector<size_t> &ids, Automaton *automaton) {
        for (size_t i = 0; i < words.size(); ++i) {
            AddString(&automaton->root_, ids[i], words[i]);
        }
    }

    static void AddString(AutomatonNode *root, size_t string_id,
                          const std::string &string);

    static void BuildSuffixLinks(Automaton *automaton);

    static void BuildTerminalLinks(Automaton *automaton);

    std::vector<std::string> words_;
    std::vector<size_t> ids_;
};

} // namespace aho_corasick

// Consecutive delimiters are not grouped together and are deemed
// to delimit empty strings
template <class Predicate>
std::vector<std::string> Split(const std::string &string,
                               Predicate is_delimiter);

// Wildcard is a character that may be substituted
// for any of all possible characters.
class WildcardMatcher {
public:
    WildcardMatcher() : number_of_words_(0), pattern_length_(0) {}

    WildcardMatcher static BuildFor(const std::string &pattern, char wildcard);

    // Resets the matcher. Call allows to abandon all data which was already
    // scanned,
    // a new stream can be scanned afterwards.
    void Reset();

    /* В данном случае Callback -- это функция,
     * которая будет вызвана при наступлении
     * события "суффикс совпал с шаблоном".
     * Почему мы выбрали именно этот способ сообщить
     * об этом событии? Можно рассмотреть альтернативы:
     * вернуть bool из Scan, принять итератор и записать
     * в него значение. В первом случае, значение bool,
     * возвращенное из Scan, будет иметь непонятный
     * смысл. True -- в смысле все считалось успешно?
     * True -- произошло совпадение? В случае итератора,
     * совершенно не ясно, какое значение туда  записывать
     * (подошедший суффикс, true, ...?). Более того, обычно,
     * если при сканировании потока мы наткнулись на
     * совпадение, то нам нужно как-то на это отреагировать,
     * это действие и есть наш Callback on_match.
     */
    template <class Callback>
    void Scan(char character, Callback on_match);

private:
    void UpdateWordOccurrencesCounters();

    void ShiftWordOccurrencesCounters();

    // Storing only O(|pattern|) elements allows us
    // to consume only O(|pattern|) memory for matcher.
    std::deque<size_t> words_occurrences_by_position_;
    aho_corasick::NodeReference state_;
    size_t number_of_words_;
    size_t pattern_length_;
    std::unique_ptr<aho_corasick::Automaton> aho_corasick_automaton_;
};

std::string ReadString(std::istream &input_stream);

// Returns positions of the first character of an every match.
std::vector<size_t> FindFuzzyMatches(const std::string &pattern_with_wildcards,
                                     const std::string &text, char wildcard);

void Print(const std::vector<size_t> &sequence);
