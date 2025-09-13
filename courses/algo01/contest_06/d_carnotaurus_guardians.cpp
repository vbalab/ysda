#include <bits/stdc++.h>
using namespace std;

struct LCT {
    struct Node {
        Node *ch[2] = {nullptr, nullptr}, *f = nullptr;
        bool rev = false;
        int id; // store node's ID for reference
    }*nodes;

    LCT(int n) {
        nodes = new Node[n+1];
        for (int i = 1; i <= n; i++)
            nodes[i].id = i;
    }

    bool isRoot(Node* x) {
        return !x->f || (x->f->ch[0] != x && x->f->ch[1] != x);
    }
    void pushDown(Node *x) {
        if (x && x->rev) {
            x->rev = false;
            if (x->ch[0]) x->ch[0]->rev = !x->ch[0]->rev;
            if (x->ch[1]) x->ch[1]->rev = !x->ch[1]->rev;
            swap(x->ch[0], x->ch[1]);
        }
    }
    void rotate(Node *x) {
        Node *f = x->f; 
        bool isR = (f->ch[1] == x);
        if(!isRoot(f)) (f->f->ch[f->f->ch[1]==f] = x);
        x->f = f->f;
        f->ch[isR] = x->ch[!isR]; 
        if(x->ch[!isR]) x->ch[!isR]->f = f;
        x->ch[!isR] = f; 
        f->f = x;
    }
    void splay(Node *x) {
        static Node* st[200001];
        int top=0; st[++top]=x;
        for(Node*y=x;!isRoot(y);y=y->f) st[++top]=y->f;
        for(;top;top--) pushDown(st[top]);
        while(!isRoot(x)) {
            Node *f=x->f;
            if(!isRoot(f))
                rotate((f->f->ch[1]==f)==(f->ch[1]==x)?f:x);
            rotate(x);
        }
        pushDown(x);
    }
    void access(Node *x) {
        Node *pre = nullptr;
        for(Node*y=x; y; y=y->f) {
            splay(y);
            y->ch[1]=pre;
            pre=y;
        }
        splay(x);
    }
    void makeRoot(Node *x) {
        access(x);
        x->rev=!x->rev;
        pushDown(x);
    }
    Node* findRoot(Node*x) {
        access(x);
        while(x->ch[0]) {
            pushDown(x);
            x=x->ch[0];
        }
        splay(x);
        return x;
    }
    void link(int u,int v) {
        // make v parent of u
        Node* x=&nodes[u], *y=&nodes[v];
        makeRoot(x);
        if(findRoot(y)!=x) {
            x->f=y;
        }
    }
    void cut(int u,int v) {
        Node*x=&nodes[u],*y=&nodes[v];
        makeRoot(x);access(y);
        // now y->ch[0] = x
        if(y->ch[0]==x) {
            y->ch[0]->f=nullptr;
            y->ch[0]=nullptr;
        }
        splay(y);
    }
    int LCA(int u,int v) {
        if(u==v) return u;
        Node*x=&nodes[u],*y=&nodes[v];
        makeRoot(x);
        access(y);
        // after this, x is splayed
        splay(x);
        // The LCA is now x if x is also on the path to y. If not:
        // Actually, after makeRoot(u), access(v), u is root. The LCA is the node in x's splay tree accessible via left child chain.
        // The LCA is the node that appears on the path from v to u's root.
        // According to known LCT LCA method: After these ops, the LCA is x if u is ancestor of v. Otherwise, the LCA is
        // the deepest node in the path. The deepest node is the one currently in x's position. If x has a left child chain:
        pushDown(x);
        while(x->ch[0]){
            x= x->ch[0];
            pushDown(x);
        }
        return x->id;
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int M; cin >> M;
    // Initially, we have dinosaur #1 which is never removed.
    // We'll have at most M+1 dinosaurs.
    // Need a free ID pool:
    priority_queue<int,vector<int>,greater<>> freeIDs;
    for (int i=2;i<=M+1;i++) freeIDs.push(i);

    // We'll keep:
    // parent[i]: current parent of node i (0 if none)
    // children[i]: vector of children
    // posInParent[i]: position of i in parent's children vector for O(1) removal
    // We'll handle the dynamic tree using LCT.

    // But we must be careful: after removal and re-hanging, the structure changes.
    // Initially: parent[1]=0

    // We'll initialize LCT with M+1 nodes, only #1 used at start.
    // As nodes are born (+ v), we take from freeIDs and link in LCT.
    // On remove(- v), do the re-hanging and LCT operations.

    // We'll store queries and process them online.

    vector<int> parent(M+2,0);
    vector<vector<int>> children(M+2);
    vector<int> posInParent(M+2, -1); // position in parent's children vector

    // Initially we have node 1, no parent, no children.

    LCT lct(M+1);

    // A helper to add a child:
    auto addChild = [&](int p,int c) {
        parent[c]=p;
        posInParent[c]=(int)children[p].size();
        children[p].push_back(c);
    };

    // A helper to remove a child from its parent quickly:
    auto removeChild = [&](int c) {
        int p=parent[c];
        int pos=posInParent[c];
        int last=(int)children[p].size()-1;
        if(pos!=last){
            int lastNode=children[p][last];
            children[p][pos]=lastNode;
            posInParent[lastNode]=pos;
        }
        children[p].pop_back();
        posInParent[c]=-1;
        parent[c]=0;
    };

    // Initially we have just node 1 in LCT, no links needed.

    // Process queries
    vector<int> answers;
    answers.reserve(M); // max number of '?' queries

    int nextID = M+2; // just a large number not to mix
    // Actually we don't need nextID if we use freeIDs
    // We'll just pop from freeIDs when needed.

    for(int i=0;i<M;i++){
        char t; cin >> t;
        if(t=='+'){
            int v; cin >> v;
            // birth a new node
            int x = freeIDs.top(); freeIDs.pop();
            // Link x as child of v
            addChild(v,x);
            // LCT link:
            // make x root
            lct.makeRoot(&lct.nodes[x]);
            // make v root
            lct.makeRoot(&lct.nodes[v]);
            lct.link(x,v);
        } else if(t=='-'){
            int v; cin >> v;
            int p = parent[v];
            // We will remove v:
            // For each child c of v:
            // cut(v,c)
            // link(c,p)
            // parent[c]=p
            // move c to children[p]
            // Actually just move the entire children[v] to children[p]
            // do it carefully:
            for (auto c: children[v]) {
                // LCT operations
                lct.cut(v,c);
                // now link c to p
                lct.makeRoot(&lct.nodes[c]);
                lct.makeRoot(&lct.nodes[p]);
                lct.link(c,p);

                parent[c]=p;
                posInParent[c]=(int)children[p].size();
                children[p].push_back(c);
            }
            children[v].clear();

            // now cut(p,v)
            lct.cut(p,v);
            // remove v from p's children
            removeChild(v);
            // v is isolated now, free its ID
            freeIDs.push(v);
        } else {
            int u,v; cin >> u >> v;
            if(u==v){
                answers.push_back(u);
                continue;
            }
            // LCA query:
            int lca = lct.LCA(u,v);
            answers.push_back(lca);
        }
    }

    for (auto ans: answers) cout << ans << "\n";

    return 0;
}
