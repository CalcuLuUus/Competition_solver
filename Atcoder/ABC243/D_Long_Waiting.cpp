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
typedef pair<ll, ll> P;

#define FIN freopen("in.txt", "r", stdin);freopen("out.txt","w",stdout);
#define endl '\n'


int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int n, k;
    cin >> n >> k;
    priority_queue<P, vector<P>, greater<P> > inq;
    priority_queue<P, vector<P>, greater<P> > outq;

    vector<int> st(n), ed(n), num(n);

    for(int i = 0; i < n; i++){
        cin >> st[i] >> ed[i] >> num[i];
        inq.push(make_pair(st[i], i));
    }

    ll nowtime = 0;
    int nowpeople = 0;
    while(inq.size() + outq.size()){
        if(inq.size() && nowpeople + num[inq.top().second] <= k && (outq.empty() || inq.top().first < outq.top().first)){
            P innow = inq.top();
            if(nowtime < innow.first){
                nowtime = innow.first;
            }
            cout << nowtime << endl;
            nowpeople += num[innow.second];
            inq.pop();
            outq.push(make_pair(nowtime + ed[innow.second], innow.second));
        }
        else{
            P outnow = outq.top();
            if(nowtime < outnow.first){
                nowtime = outnow.first;
            }
            nowpeople -= num[outnow.second];
            outq.pop();
        }
    }
 
    
    return 0;
}












