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

const int mod = 1e9+7;

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

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int t;

    cin >> t;
    while(t--){
        ll n;
        cin >> n;
        vector<int> a(n+5);
        for(int i = 0; i < n; i++){
            cin >> a[i];
        }
        
        vector<vector<int> > dp(n + 1, vector<int>(n + 1, 0));
        dp[0][0] = 1;

        for (int id = 0; id < n; id++) {
            int val = a[id];
            vector<vector<int> > next_dp = dp;
            for (int i = 0; i <= n; ++i) {
                for (int j = i; j <= n; ++j) {
                    if (dp[i][j] > 0) {
                        int count = dp[i][j];
                        if (val >= j) {
                            next_dp[i][val] = (next_dp[i][val] + count) % mod;
                        } 
                        else if (val >= i) {
                            next_dp[val][j] = (next_dp[val][j] + count) % mod;
                        }
                    }
                }
            }
            dp = next_dp;
        }

        ll ans = 0;
        for (int i = 0; i <= n; ++i) {
            for (int j = 0; j <= n; ++j) {
                ans = (ans + dp[i][j]) % mod;
            }
        }

        cout << ans << endl;
    }
    
    return 0;
}












