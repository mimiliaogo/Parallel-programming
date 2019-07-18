# 7/3_L2
## Supercomputer & Latest technologies
```
core & frequency has trade off
47. TPU doing matrix operation is specified for deep learning
49. 同步化 大家要有相同的資訊
50. (interconnection : logical connection) + physical connection(cable) 
51. 
latency: the worst case is the distance btw begin and end
Bisection: 要砍幾刀才會斷掉 (resilience: tolerance to fail)
53. Torus 就是mesh 前後連一起
58. infiniBand reduce memory copy
RDMA (remote) ( direct memory access) 
63. IO parallel -> 資料儲存在不同電腦

```
## Parallel Program Analysis
```
using Speedup with Strong scalability

communication or IO overhead make the linear down

72.strong scalibility : size remains, processing elements increases
weak : 計算improvement increases, communication time also so maybe 抵銷，但strong 計算量是一樣的，communication increases so it's harder to acheive ideal.
```
## MPI
```

6. MPI 讓在每一個機器上跑相同的process 透過send receive communicate
8. 使用者決定用哪種方式溝通
by溝通方式 : 
    1. 同時送出與接受
    2. 非同步 ex. email
    blocking code: 等他做完 再下一行
    non-blocking code: return means it scheduling to do it
    *not neccessarily one to one
10. 有buffer so don't need to wait 
if recv 先被called , there needs to be a checked function to avoid getting  incorrect values.
```
```
12. return error code and pass value by reference (not return value)
init finalize中間夾平行program(call MPI functions)
13. communicator 告訴是哪個group 
rank 針對each group裡面

14. 多少人: MPI_Comm_size(comm, &size)
一開始comm傳comm_world
我是誰: MPI_Comm_rank(comm, &rank)

17. request 用來看當初的request
19. Isend 完可以直接compute recv那邊繼續receive 比blocking 快
21. Barrier:同步化
collective call 幾乎就是blocking
確保大家停在同一個點

盡量用API. turn to MPI reference
新的group 新的communicator
```