//C++ style std::thread
#include <thread>
#include <iostream>
void aplusb(int a, int b, int *c)
{
    *c = a + b;
}
int main()
{
    int c;
    std::thread th(aplusb, 1, 2, &c);
    th.join();
    std::cout<< c << '\n';
}