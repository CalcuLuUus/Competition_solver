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

int dx[] = {0, 1, 0, -1};
int dy[] = {1, 0, -1, 0};
const int MAXN = 55;
int a[MAXN][MAXN];

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int n, m;
    cin >> n >> m;
    
    for(int i = 0; i < n; i++){
    	for(int  j =0; j < m; j++){
    		for(int k = 0; k < 4; k++){
    			int nx = i + dx[k], ny = j + dy[k];
    			if(nx >= 0 && nx < n && ny >= 0 && ny < m){
    				a[i][j] ++;
				}
			}
		}
	}
	
	for(int i = 0; i < n; i++){
		for(int j  = 0; j <  m; j++){
			cout << a[i][j] << ' ';
		}
		cout << endl;
	}
   	



    return 0;
}












