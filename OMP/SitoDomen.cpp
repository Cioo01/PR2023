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
#define LIMIT 2000000000

using namespace std;

int threads = omp_get_max_threads();
double start_t, stop_t;

void displayResults(vector <int> primes)
{
    cout << "Primes: " << endl;

    // for(int prime : primes)
    //     cout<<prime<<" ";

    cout << endl << "Found " << primes.size() << " primes" << endl;
}

int** divide(int lowerLimit, int upperLimit)
{
    int** output = new int* [threads];
    for (int i = 0; i < threads; i++)
        output[i] = new int[2];
    int intsPerSet = (upperLimit - lowerLimit) / threads;
    for (int i = 1; i <= threads; i++)
        if (i == threads)
        {
            output[i - 1][0] = lowerLimit + intsPerSet * (i - 1);
            output[i - 1][1] = upperLimit;
        }
        else
        {
            output[i - 1][0] = lowerLimit + intsPerSet * (i - 1);
            output[i - 1][1] = lowerLimit + intsPerSet * i - 1;
        }
    return output;
}

void basicSieve(int upper, vector<int>& input)
{
    bool* primes = new bool[upper - 1];
    for (int i = 0; i <= upper - 2; i++)
        primes[i] = PRIME;

    for (int p = 2; p * p <= upper; p++) {
        if (primes[p - 2] == true) {
            for (int i = p * p; i <= upper; i += p)
                primes[i - 2] = false;
        }
    }

    for (int i = 0; i <= upper - 2; i++)
        if (primes[i] == PRIME) {
            //cout << i + 2 << endl;
            input.push_back(i + 2);
        }
    delete primes;
}

vector<int> domain(int lower, int upper)
{
    vector<int> primes;
    int** subsets = divide(lower, upper);
    vector <int> primesToSqrt;
    vector <vector<bool>> subsetOutput(threads);

    start_t = omp_get_wtime();
    basicSieve(sqrt(upper), primesToSqrt);

#pragma omp	parallel
    {
        int threadNum = omp_get_thread_num();

        vector<bool> subset(subsets[threadNum][1] - subsets[threadNum][0] + 1, PRIME);

        for (int i = 0; i < primesToSqrt.size(); i++)
        {
            int sieved = primesToSqrt[i];
            int number = subsets[threadNum][0];
            for (; number % sieved != 0; number++)
                continue;
            if (number == sieved)
                number *= 2;

            for (; number <= subsets[threadNum][1]; number += sieved) {
                subset[number - subsets[threadNum][0]] = COMPLEX;
            }
        }
        subsetOutput[threadNum] = subset;
    }

    stop_t = omp_get_wtime();

    /*for (int i = 0; i < threads; i++)
    {
        for (int j = 0; j < subsetOutput[i].size(); j++)
        {
            if (subsetOutput[i][j] == PRIME)
            {
                primes.push_back(subsets[i][0] + j);
            }
        }
    }*/

    cout << "Processing " << upper - lower << " took: " << stop_t - start_t << " sec" << endl;

    return primes;
}

int main()
{
    vector<int> result = domain(2, B1);
    //displayResults(result);
}