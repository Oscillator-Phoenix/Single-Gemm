#ifndef __USE_TIMER_H__
#define __USE_TIMER_H__

#include <stdio.h>
#include <stdlib.h>

// abtic() returns the time in nanoseconds
#if defined(_WIN32) && defined(_MSC_VER)
#include <windows.h>
double abtic()
{
    __int64 freq;
    __int64 clock;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&clock);
    return ((double)clock / freq) * 1000 * 1000 * 1000;
}
#else
#include <sys/time.h>
#include <time.h>

double abtic()
{
    double result = 0.0;
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1)
    {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    result = ts.tv_sec * 1000 * 1000 * 1000 + ts.tv_nsec;
    return result;
}
#endif // _WIN32

#if 1
double timer_s;
double timer_e;
#define ABTMS(msg)                                                     \
    do                                                                 \
    {                                                                  \
        fprintf(stdout, "%s:%4d %s begin\n", __FILE__, __LINE__, msg); \
        timer_s = abtic();                                             \
    } while (0)

#define ABTME(msg)                                                                                               \
    do                                                                                                           \
    {                                                                                                            \
        timer_e = abtic();                                                                                       \
        fprintf(stdout, "%s:%4d %s end   %8.8fms\n", __FILE__, __LINE__, msg, (timer_e - timer_s) / 1000000.0f); \
    } while (0)
#else
#define ABTMS
#define ABTME
#endif // 1

#endif // __USE_TIMER_H__
