## L3_7/4
```
32.can't write it simultaneously
each P open the same file, MPI let the process fopen the file in order.(for file system 一次一個)
33. Collective: 每個人都要call
buffer for the maximun btw each .
one fread() request

```
### Pthread
```
thread vs. process
process doesn't have shared memory 

Program running -> process (CPU can access)

4. thread: 保留住gloabl memory (共用)
每一個thread 執行一個function
每一個function has local variable(stack: duplicate) register(instruction counter自己決定執行到哪)
5. fork(): create process 

8. pthread_create() OS create a funciton (function return -> tread end) 
various argument->struct
9. global variable 大家都可以看到
你會發現，印出來的值 有可能2 2 3 3...因為沒有synchronize
main->main thread
10. 會一直等join 所以如果要告訴 這個thread 不用return call detach
choose to call join or detach
13. result depends on the order (avoid)

16. 第一個搶先進來就lock起來了
直到執行完critical section
but CPU busy waiting

17. init 完 就可用mutex
通常有write variable 就要加lock unlock
要加在所有會搶相同的global variable 的function

18. producer 從in 開始寫
consumer 從out 開始讀
in=out->empty
full-> 犧牲一格 or empty = full

23.counter: to wake up others (signal() / broadcast())
pthread_cond_wait(cond, mutex)先做unlock counter才不會被綁住
why use lock?
A: 在判斷x!=0期間可能counter x會改變
```
### OpenMP
```
比較像MPI 
每個thread 做的是差不多
#pragma....
裡面的東西不能跳出去

8. thread create thread = 6 threads

11. 
do/for: data parallelism -> like matrix, array...
sections: do diff functions
single: like IO

12. schedule 怎麼分配thread 要各自要做甚麼
ordered 保證output 跟sequential output 一樣(但效能很差避免)
Static:
chunks -> 排程工作的單位 (一條thread做多少)
Dynamic:(工作量不一致時再用)
先做完自己的chunks，去搶下一件事情

23. 
critical: like lock and unlock 
atomic : 保證 shared memory access 一次取完

27. btween thread 哪些 是 global local memory 

29. private : 告訴compiler 把global declare 成 each thread 的 private variable 
firstprivate: 如果用private print出來會是亂碼
，firstprivate has initial
lastprivate: if private, var1 在region change will not affect the global value
but lastprivate will reserve for the last access in thread

30. default: all variables are the same
REDUCTION: copy the local variables into global use some operations



```