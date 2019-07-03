# Parallel progamming
### 7/2_L1
```
sequential process ->一次一個instruction
parallel->instruction 同時執行 變快 多個core執行

vs. distributed computing (internet) 分散式系統
分散在很多地方，想要彼此共享 不一定效能要很快


深度學習 tune 參數 用平行計算快多了 用GPU

10. 跨電腦 pointer 取到不同的memory address(vs.muti-core)
using MPI (Message passing interface)
用平行的框架換成底層的平行計算
CUDA low level (higher performance)
Pthread low level
OpenMP higher level ->automatically produce pthread(lower performance)
```

#### Flynn’s classic taxonomy
```
11. 所有processor可以吃甚麼?
13. Like sigle core
14. ex. 十個各load  一個資料 (like array) 很多 processor ex. GPU 
15.現實不存在 每個processor 吃相同data 不同instruction 沒有這種計算邏輯 通常用來做fault tolerance 吃相同instruction 相同data 看誰先出來就用誰的，如果有人fail 也沒關係
16. typical CPU most flexible
hardware limitation 不能太多PU
```
#### Memory architecture classification
```
18.
Distributed memory 跨電腦
好處: 各自管理，值不會修到彼此的
只會copy no overide ex. MPI
Shared memory 同台電腦
ex. Pthread

20. Ways of memory access
UMA for small chip
NUMA ex. super computer (too many CPU)
21. Cluster :(small scale ) 單一管理系統
Supercomputer: 科學計算
Datacenter: large scale ex. Google 
22.明確的知道 值被改了
send and receive
```
#### Programming model classification
```
抽象硬體
25. 一個程式在跑由memory content 組成 不同process memory content is independent but thread programming (create thread) let the three content (code, data, files) 因為main thread 已經把code load 進來。
透過global variable 溝通
node ->一台電腦
```