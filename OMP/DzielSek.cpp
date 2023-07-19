#include <iostream>
#include <math.h>
#include <time.h>

#define START 2
#define LIMIT 1000
#define PRIME true;
#define COMPLEX false;
using namespace std;

bool CheckIfPrime(int number)
{
    for(int i=2;i<=sqrt(number);i++)
        if(number%i == 0)
            return COMPLEX;
    return PRIME;
}

int main()
{
    clock_t start, stop;
    int count = 0;

    start= clock();

    for(int i=START;i<LIMIT;i++)
        if(CheckIfPrime(i))
            count++;

    stop = clock();

    cout<<endl<<"Found "<<count<<" primes"<<endl;
    cout<<"Processing "<<LIMIT<<" took: "<<(double)(stop-start)/CLOCKS_PER_SEC << " sec"<< endl;
}