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

    int t;
    cin >> t;
    while(t--){
        int n;
        cin >> n;
        vector<int> a(n);
        for(int i = 0; i < n; i++){
            cin >> a[i];
        }

        int l = 0, r = n-1;
        int fail = 0;
        for(int k = n; k >= 1; k--){
            for(int i = l; i <= r; i++){
                a[i]--;
            }
            for(int i = l; i <= r; i++){
                if(a[i] < 0){
                    fail = 1;
                    break;
                }
            }
            if(a[l] == 0) l++;
            else if(a[r] == 0) r--;
            else{
                fail = 1;
            }
            if(fail){
                break;
            }
        }
        if(fail){
            cout << "NO" << endl;
        }
        else{
            for(int i = 0; i < n; i++){
                if(a[i] != 0){
                    fail = 1;
                    break;
                }
            }
            if(fail){
                cout << "NO" << endl;
            }
            else{
                cout << "YES" << endl;
            }
        }

    }
    
    return 0;
}












