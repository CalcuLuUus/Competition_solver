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

    int n, q;
    cin >> n >> q;
    vector<ll> a(n+5), pa(n+5), p2a(n+5);
    for(int i = 1; i <= n; i++){
        cin >> a[i];
        pa[i] = a[i] * i;
        p2a[i] = a[i] * i * i;
    }
    vector<ll> suma(n+5), sumpa(n+5), sump2a(n+5);
    for(int i = 1; i <= n; i++){
        suma[i] = suma[i-1] + a[i];
        sumpa[i] = sumpa[i-1] + pa[i];
        sump2a[i] = sump2a[i-1] + p2a[i];
    }

    while(q--){
        ll li, ri;
        cin >> li >> ri;
        ll ans = 0;
        ans -= (li - 1)*(ri+1)*(suma[ri] - suma[li-1]);
        ans -= sump2a[ri] - sump2a[li-1];
        ans += (li + ri) * (sumpa[ri] - sumpa[li-1]);
        cout << ans << endl;
    }


    
    return 0;
}












