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
    int n, r;
    cin >> n >> r;
    vector<int> a(n+5);
    int flg = 0;
    for(int i = 1; i <= n; i++){
        cin >> a[i];
        if(a[i] == 0) flg = 1;
    }
    if(flg == 0){
        cout << 0 << endl;
        return 0;
    }
    int L = 1, R = n;
    while(a[L] && L <= n && L < R) L++;
    while(a[R] && R >= 1 && L < R) R--;

    if(r < L){
        L = r+1;
    }
    if(R < r){
        R = r;
    }
    int ans = 0;
    for(int i = L; i <= R; i++){
        if(a[i] == 0){
            ans++;
        }
        else{
            ans += 2;
        }
    }
    cout << ans << endl;
    
    return 0;
}












