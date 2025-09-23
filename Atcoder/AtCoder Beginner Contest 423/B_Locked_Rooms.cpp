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


int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int n;
    cin >> n;
    vector<int> a(n+5), vis(n+5);
    for(int i = 1; i <= n; i++){
        cin >> a[i];
    }
    int l = 0, r = n;
    vis[l] = 1;
    vis[r] = 1;
    while(a[l+1] == 0){
        l++;
        vis[l] = 1;
        if(l == n) break;
    }
    while(a[r] == 0){
        r--;
        vis[r] = 1;
        if(r == 0) break;
    }
    int ans = 0;
    for(int i = 1; i <= n; i++){
        ans += vis[i];
    }
    cout << n - ans << endl;
    
    return 0;
}












