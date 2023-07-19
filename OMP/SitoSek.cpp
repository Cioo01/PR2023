#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>

#define START 50000
#define LIMIT 100000
#define PRIME true
#define COMPLEX false

using namespace std;

bool* sieve(int start, int end)
{
    clock_t start_t, stop_t;
    bool* primes = new bool[end+1];
    for(int i=0;i<=end;i++)
        *(primes+i) = PRIME;
    primes[0] = primes[1] = false;
    start_t = clock();
    int limit = sqrt(end);
    for (int i = 2; i <= limit; i++)
        if (*(primes+i))
            for (int j = i * i; j <= end; j += i)
                *(primes+j) = false;
    stop_t = clock();
    cout << "Processing " << START << " to " << LIMIT << " took: " << (double)(stop_t - start_t) / CLOCKS_PER_SEC << "s" << endl;
    return primes;
}

int main()
{
    int count = 0;
    bool* primes = sieve(START, LIMIT);
    
    for(int i=START;i<LIMIT;i++)
        if(*(primes+i))
            count++;
    cout << "Found " << count << " primes" << endl;
    return 0;
}