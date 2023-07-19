#include <stdio.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <math.h>

#define PRIME true
#define COMPLEX false
#define K1 1000
#define K10 10000
#define K100 100000
#define M1 1000000
#define M10 10000000
#define M100 100000000
#define B1 1000000000
#define B2 2000000000

using namespace std;

int threads = omp_get_max_threads();
double start_t, stop_t;

void displayResults(vector <int> primes)
{
 /*   cout << "Primes: " << endl;

     for(int prime : primes)
         cout<<prime<<" ";*/

    cout << endl << "Found " << primes.size() << " primes" << endl;
}


void basicSieve(int upper, vector<int>& input)
{
    bool* primes = new bool[upper - 1];
    for (int i = 0; i <= upper - 2; i++)
        primes[i] = PRIME;
    for (int p = 2; p * p <= upper; p++) {
        if (primes[p-2] == true) {
            for (int i = p * p; i <= upper; i += p)
                primes[i-2] = false;
        }
    }

    for (int i = 0; i <= upper - 2; i++)
        if (primes[i] == PRIME) {
            //cout << i + 2 << endl;
            input.push_back(i + 2);
        }
    delete primes;
}

vector<int> function(int lower, int upper)
{
    vector<int> primes;
    vector <int> primesToSqrt;
    vector <vector<bool>> primeInRange(threads);
    start_t = omp_get_wtime();
    basicSieve(sqrt(upper), primesToSqrt);

#pragma omp	parallel
    {
        vector<bool> localIsPrime(upper - lower + 1, PRIME);

#pragma omp	for schedule(dynamic)
        for (int i = 0; i < primesToSqrt.size(); i++)
        {
            int sieved = primesToSqrt[i];
            int number = lower;
            for (; number % sieved != 0; number++)
                continue;
            if (number == sieved)
                number *= 2;

            for (; number <= upper; number += sieved)
                localIsPrime[number - lower] = COMPLEX;
        }
        primeInRange[omp_get_thread_num()] = localIsPrime;
    }

    stop_t = omp_get_wtime();

    vector <bool> primesMerge(2, COMPLEX);
    bool flag;
    /*for (int i = 0; i < upper - lower + 1; i++)
    {
        flag = true;
        for (int j = 0; j < threads; j++)
            flag = flag && primeInRange[j][i];
        primesMerge.push_back(flag);
    }

    for (int i = 0; i < primesMerge.size(); i++)
        if (primesMerge[i] == PRIME)
            primes.push_back(i);*/
    cout << "Processing " << upper - lower << " took: " << stop_t - start_t << " sec" << endl;
    return primes;
}

int main()
{
    vector<int> result = function(2, B1);
    displayResults(result);
    //vector<int> result;
    //basicSieve(1000, result);
}