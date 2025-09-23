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

bool cmp(int a, int b){
    return a > b;
}

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int t;
    cin >> t;
    while(t--){
        int n, k;
        cin >> n >> k;
        vector<int> a(n), val(k);
        for(int i = 0; i < n; i++){
            cin >> a[i];
        }
        for(int i = 0; i < k; i++){
            cin >> val[i];
        }

        sort(a.begin(), a.end(), cmp);
        sort(val.begin(), val.end());

        int cur = 0;
        int vcur = 0;
        ll ans = 0;
        while(cur < n && vcur < k){
            int numval = val[vcur];
            vcur++;
            while(numval > 1 && cur < n){
                ans = ans + a[cur];
                cur++;
                numval--;
            }
            cur++;
        }
        while(cur < n){
            ans = ans + a[cur];
            cur++;
        }

        cout << ans << endl;

        
    }
    
    return 0;
}












