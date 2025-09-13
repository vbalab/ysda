#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

class DisjointSetUnion {
public:
    std::vector<int> parent_array;
    std::vector<int> size_array;

    DisjointSetUnion(int num_elements) {
        parent_array.resize(num_elements + 1);
        size_array.resize(num_elements + 1, 1);
        for (int index = 1; index <= num_elements; index++) {
            parent_array[index] = index;
        }
    }

    int FindSet(int employee) {
        if (parent_array[employee] == employee) {
            return employee;
        }
        parent_array[employee] = FindSet(parent_array[employee]);
        return parent_array[employee];
    }

    bool Union(int employee_a, int employee_b) {
        employee_a = FindSet(employee_a);
        employee_b = FindSet(employee_b);
        if (employee_a == employee_b) {
            return false;
        }
        if (size_array[employee_a] < size_array[employee_b]) {
            std::swap(employee_a, employee_b);
        }
        parent_array[employee_b] = employee_a;
        size_array[employee_a] += size_array[employee_b];
        return true;
    }
};

int FindBoss(std::vector<int>& direct_boss_list, int employee_x) {
    if (direct_boss_list[employee_x] == 0) {
        return employee_x;
    }
    direct_boss_list[employee_x] =
        FindBoss(direct_boss_list, direct_boss_list[employee_x]);
    return direct_boss_list[employee_x];
}

void ProcessQueries(int num_queries, DisjointSetUnion& dsu,
                    std::vector<int>& direct_boss_list) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    for (int query_index = 0; query_index < num_queries; query_index++) {
        std::string line;
        if (!std::getline(std::cin, line)) {
            if (!std::getline(std::cin, line)) {
                continue;
            }
        }
        if (line.empty()) {
            query_index--;
            continue;
        }

        std::stringstream ss(line);
        std::vector<int> values;
        int read_value;
        while (ss >> read_value) {
            values.push_back(read_value);
            if (values.size() == 2) {
                break;
            }
        }

        if (values.size() == 2) {
            int employee_a = values[0];
            int employee_b = values[1];
            if (direct_boss_list[employee_b] == 0) {
                int root_a = dsu.FindSet(employee_a);
                int root_b = dsu.FindSet(employee_b);
                if (root_a != root_b) {
                    direct_boss_list[employee_b] = employee_a;
                    dsu.Union(employee_a, employee_b);
                    std::cout << 1 << "\n";
                } else {
                    std::cout << 0 << "\n";
                }
            } else {
                std::cout << 0 << "\n";
            }
        } else {
            int employee_a = values[0];
            int boss = FindBoss(direct_boss_list, employee_a);
            std::cout << boss << "\n";
        }
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int num_employees;
    int num_queries;

    std::cin >> num_employees;
    std::cin >> num_queries;

    std::vector<int> direct_boss_list(num_employees + 1, 0);
    DisjointSetUnion dsu(num_employees);

    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    ProcessQueries(num_queries, dsu, direct_boss_list);

    return 0;
}
