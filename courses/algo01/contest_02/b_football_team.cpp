#include <cstdint>
#include <iostream>
#include <vector>

template <typename Iterator, typename Comparator>
void InsertionSort(Iterator begin, Iterator end, Comparator comp) {
    for (Iterator i = begin + 1; i != end; ++i) {
        auto key = *i;
        Iterator lower = i;

        while (lower != begin && comp(key, *(lower - 1))) {
            *lower = *(lower - 1);
            --lower;
        }
        *lower = key;
    }
}

template <typename Iterator, typename Comparator>
Iterator MedianOfFives(Iterator begin, Iterator end, Comparator comp) {
    if (end - begin <= 5) {
        InsertionSort(begin, end, comp);

        return begin + (end - begin) / 2;
    }

    Iterator iter = begin;
    for (Iterator group_begin = begin; group_begin < end; group_begin += 5) {
        Iterator group_end = group_begin + 5;
        if (group_end > end) {
            group_end = end;
        }

        InsertionSort(group_begin, group_end, comp);

        Iterator median = group_begin + (group_end - group_begin) / 2;
        std::iter_swap(iter, median);

        ++iter;
    }

    return MedianOfFives(begin, iter, comp);
}

template <typename Iterator, typename Comparator>
Iterator LomutoPartition(Iterator begin, Iterator end, Iterator pivot,
                         Comparator comp) {
    Iterator split = begin;

    std::iter_swap(pivot, (end - 1));  // trick

    for (Iterator i = begin; i != end - 1; ++i) {
        if (comp(*i, *(end - 1))) {
            std::iter_swap(split, i);
            ++split;
        }
    }

    std::iter_swap(split, (end - 1));  // trick

    return split;
}

// It's faster...
template <typename Iterator, typename Comparator>
Iterator HoarePartition(const Iterator& begin, const Iterator& end,
                        const Iterator& pivot, const Comparator& comp) {
    auto pivot_value = *pivot;

    std::iter_swap(pivot, begin);
    Iterator left = begin + 1;
    Iterator right = end - 1;

    while (left <= right) {
        while (left <= right && comp(*left, pivot_value)) {
            ++left;
        }
        while (left <= right && comp(pivot_value, *right)) {
            --right;
        }

        if (left >= right) {
            break;
        }

        std::iter_swap(left, right);
        ++left;
        --right;
    }

    std::iter_swap(right, begin);
    return right;
}

template <typename Iterator,
          typename Comparator = std::less<decltype(*std::declval<Iterator>())>>
void QuickSort(Iterator begin, Iterator end, Comparator comp = Comparator()) {
    while (end - begin > 1) {  // assumes: random access Iterator
        Iterator pivot = MedianOfFives(begin, end, comp);

        // Iterator split = LomutoPartition(begin, end, pivot, comp);
        Iterator split = HoarePartition(begin, end, pivot, comp);

        QuickSort(begin, split, comp);

        begin = split + 1;  // Tail Recursion Optimization
    }
}

struct Player {
    int64_t efficiency;
    std::size_t number;

    Player() {}

    Player(const int64_t& efficiency, const std::size_t& number)
        : efficiency(efficiency), number(number) {}
};

bool EfficiencyComparator(const Player& p1, const Player& p2) {
    return p1.efficiency < p2.efficiency;
}

bool NumberComparator(const Player& p1, const Player& p2) {
    return p1.number < p2.number;
}

template <typename Iterator>
struct Team {
    int64_t efficiency;
    Iterator begin;
    Iterator end;

    Team(int64_t efficiency, const Iterator& begin, const Iterator& end)
        : efficiency(efficiency), begin(begin), end(end) {}
};

std::vector<Player> BuildMostEffectiveSolidaryTeam(
    std::vector<Player> players) {
    if (players.size() <= 2) {
        return players;
    }

    QuickSort(players.begin(), players.end(), EfficiencyComparator);

    Team team = Team(0, players.begin(), players.begin());
    Team best_team = Team(0, players.begin(), players.begin());

    int64_t allowed = players[0].efficiency + players[1].efficiency;

    while (team.end != players.end()) {
        while (team.end >= team.begin + 2 && team.end->efficiency > allowed) {
            team.efficiency -= team.begin->efficiency;
            ++team.begin;

            allowed = team.begin->efficiency + (team.begin + 1)->efficiency;
        }

        team.efficiency += team.end->efficiency;

        ++team.end;

        if (team.efficiency > best_team.efficiency) {
            best_team = team;
        }
    }

    return {best_team.begin, best_team.end};
}

int64_t CalculateTeamEfficiency(const std::vector<Player>& team) {
    int64_t efficiency = 0;
    for (const Player& player : team) {
        efficiency += player.efficiency;
    }

    return efficiency;
}

void InputData(std::vector<Player>& players) {
    std::size_t n_players;
    std::cin >> n_players;

    players.resize(n_players);

    for (std::size_t i = 0; i < n_players; ++i) {
        std::cin >> players[i].efficiency;
        players[i].number = i + 1;
    }
}

void OutputData(std::vector<Player>& players) {
    QuickSort(players.begin(), players.end(), NumberComparator);

    int64_t best_eff = CalculateTeamEfficiency(players);
    std::cout << best_eff << '\n';

    for (const Player& player : players) {
        std::cout << player.number << ' ';
    }
    std::cout << '\n';
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::vector<Player> players;
    InputData(players);

    std::vector<Player> best_team = BuildMostEffectiveSolidaryTeam(players);

    OutputData(best_team);

    return 0;
}
