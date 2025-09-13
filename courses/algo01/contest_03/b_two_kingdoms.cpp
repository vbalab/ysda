#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class TreeCentroid {
public:
    TreeCentroid(int n) : n(n), adj(n), subtree_size(n), centroids() {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> find_centroids() {
        dfs(0, -1);

        return centroids;
    }

private:
    int n;
    vector<vector<int>> adj;        // adjacency list
    vector<int> subtree_size;       // size of the subtree rooted at each node
    vector<int> centroids;          // store centroids

    // DFS to compute subtree sizes and identify centroids
    void dfs(int node, int parent) {
        subtree_size[node] = 1;
        bool is_centroid = true;

        for (int neighbor : adj[node]) {
            if (neighbor != parent) {
                dfs(neighbor, node);
                subtree_size[node] += subtree_size[neighbor];

                // If any subtree is larger than half the size of the whole tree, this node is not a centroid
                if (subtree_size[neighbor] > n / 2) {
                    is_centroid = false;
                }
            }
        }

        // The size of the "complement" subtree (tree formed by removing the current node)
        int complement_size = n - subtree_size[node];
        if (complement_size > n / 2) {
            is_centroid = false;
        }

        // If the node satisfies the conditions, it's a centroid
        if (is_centroid) {
            centroids.push_back(node);
        }
    }
};

int main() {
    int n;
    cout << "Enter number of nodes in the tree: ";
    cin >> n;

    TreeCentroid tree(n);

    cout << "Enter the edges (u v): " << endl;
    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        tree.add_edge(u, v);
    }

    vector<int> centroids = tree.find_centroids();
    
    cout << "Centroid(s): ";
    for (int centroid : centroids) {
        cout << centroid << " ";
    }
    cout << endl;

    return 0;
}
