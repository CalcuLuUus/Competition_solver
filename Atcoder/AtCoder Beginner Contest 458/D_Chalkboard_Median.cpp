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



int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    priority_queue<int> up;
    priority_queue<int, vector<int>, greater<int> > down;
    
    int x, q;
    cin >> x >> q;
    
    up.push(x);
    while(q--){
    	int t = 2;
    	while(t--){
    		int num;
    		cin >> num;
    		if(num <= up.top()){
    			up.push(num);
			}
			else{
				down.push(num);
			}
			
			if(up.size() - down.size() == 2){
				int tmp = up.top();
				up.pop();
				down.push(tmp);
			}
			
			if(down.size() - up.size() == 1){
				up.push(down.top());
				down.pop();
			}
		}
		cout << up.top() << endl;;
	}



    return 0;
}












