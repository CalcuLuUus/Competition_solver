#include <iostream>
#include <cstdio>
#include <set>
#include <list>
#include <vector>
#include <stack>
#include <queue>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
//#include <unordered_map>
using namespace std;
#define dbg(x) cerr << #x " = " << x << endl;
typedef long long ll;
typedef pair<int, int> P;

#define FIN freopen("in.txt", "r", stdin);freopen("out.txt","w",stdout);
#define endl '\n'

const int MAXN = 2e5+5;

struct Node{
    int u, v;
    P xy;
};

bool cmp(Node a, Node b){
    return max(a.xy.first, a.xy.second) > max(b.xy.first, b.xy.second);
}

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int t;
    cin >> t;
    while(t--){
        int n;
        cin >> n;
        vector<Node> nodes(n);
        for(int i = 0; i < n-1; i++){
            cin >> nodes[i].u >> nodes[i].v;
            cin >> nodes[i].xy.first >> nodes[i].xy.second;
        }

        sort(nodes.begin(), nodes.end(), cmp);

        vector<vector<int> > g(n+5, vector<int>());
        vector<int> in(n+5, 0);

        for(int i = 0; i < n-1; i++){
            Node now = nodes[i];
            int u = now.u;
            int v = now.v;
            int x = now.xy.first;
            int y = now.xy.second;
            if(x > y){
                g[u].push_back(v);
                in[v]++;
            }
            else{
                g[v].push_back(u);
                in[u]++;
            }
        }

        queue<int> q;
        for(int i = 1; i <= n; i++){
            if(in[i] == 0){
                q.push(i);
            }
        }

        vector<int> ans(n+5, 0);
        int nowval = n;

        while(!q.empty()){
            int now = q.front();
            q.pop();
            ans[now] = nowval;
            nowval--;
            for(int i = 0; i < g[now].size(); i++){
                int v = g[now][i];
                in[v]--;
                if(in[v] == 0){
                    q.push(v);
                }
            }
        }

        for(int i = 1; i <= n; i++){
            cout << ans[i] << " ";
        }
        cout << endl;
        

        

    }

    
    return 0;
}












