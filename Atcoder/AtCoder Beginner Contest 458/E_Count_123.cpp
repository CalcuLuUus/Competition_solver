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


const int mod = 998244353;

const int MAXN = 2e6+5;

ll qpow(ll a, ll b){
    ll ans = 1, base = a;
    while(b){
        if(b & 1){
            ans = ans * base % mod;
        }
        base = base * base % mod;
        b >>= 1;
    }
    return ans;
}

ll inv(ll x){
    return qpow(x, mod - 2);
}

ll fac[MAXN], invfac[MAXN];

void init(){
    fac[0] = 1;
    for(int i = 1; i < MAXN; i++){
        fac[i] = fac[i-1] * i % mod;
    }
    invfac[MAXN-1] = inv(fac[MAXN-1]);
    for(int i = MAXN - 2; i >= 0; i--){
        invfac[i] = invfac[i+1] * (i + 1) % mod;
    }
}

ll C(int n, int m){
    if(m > n) return 0;
    if(n < 0 || m < 0) return 0;
    ll ans = fac[n] * invfac[m] % mod * invfac[n-m] % mod;
    return ans;
}

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    
    init();
    int x1, x2, x3;
    cin >> x1 >> x2 >> x3;
    ll ans = 0;

    for(int i = 1; i <= min(x1, x2+1); i++){
        ll tmp = C(x2+1, i) * C(x1-1, i-1) % mod;
        if(i == x2 + 1){
            if(x3 != 0) tmp = 0;
        }
        else{
            tmp = tmp * C(x2+x3-i, x2-i) % mod;
        }
        ans = (ans + tmp) % mod;
    }

    cout << ans << endl;

    return 0;
}












