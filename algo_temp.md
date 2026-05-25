# Math

## qpow & inv & C(n, m)

```cpp
const int mod = 998244353;

const int MAXN = 1e6+5;

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
```