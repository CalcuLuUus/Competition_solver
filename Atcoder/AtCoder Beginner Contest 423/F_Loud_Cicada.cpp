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
typedef __int128 LL;
typedef pair<int, int> P;

#define FIN freopen("in.txt", "r", stdin);freopen("out.txt","w",stdout);
#define endl '\n'

LL C[25][25], sum[25][25];

LL gcd(LL a, LL b){
    return b ? gcd(b, a % b) : a;
}

LL lcm(LL a, LL b){
    return a / gcd(a, b) * b;
}


int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int n, m;
    ll y;
    cin >> n >> m >> y;
    vector<ll> a(n);

    for(int i = 0; i < n; i++){
        cin >> a[i];
    }
    for(int i = 0; i <= n; i++){
        C[i][0] = C[i][i] = 1;
        for(int j = 1; j < i; j++){
            C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
        }
    }

    LL ans = 0;

    for(int st = 0; st < (1 << n); st++){
        int k = __builtin_popcount(st);
        if(k < m) continue;

        LL lcm_val = 1;
        int flg = 0;
        for(int i = 0; i < n; i++){
            if(st & (1 << i)){
                lcm_val = lcm(lcm_val, a[i]);
                if(lcm_val > y){
                    flg = 1;
                    break;
                }
            }
        }

        if(flg) continue;

        if((k - m) % 2 == 0){
            ans += C[k][m] * (y / lcm_val);
        }else{
            ans -= C[k][m] * (y / lcm_val);
        }

    }

    cout << (ll)ans << endl;

    
    return 0;
}












