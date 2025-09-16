## 用户行为预测

### Solution 1

baseline， 全局最优

直接将Top30物品作为所有人的答案，

rk70/91，分数0.00669

### Solution 2

baseline, Item-CF

统计数据对（A，B）的数量，算出最关联的A->B，然后贪心根据irank2的itemid取得到irank1的的itemid

若不满足30数量的则用Solution1作为补充

rank37， 分数0.09581

### Solution 3

baseline，传统Item-CF

计算用户-物品矩阵，利用不同用户的购买共性计算物品相似度，然后加权算答案

忽略的序列的要素

分数0.00569


---

### Solution 4

Lightgbm

rank:29 分数0.10702
