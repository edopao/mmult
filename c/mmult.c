#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gperftools/profiler.h"

#define M 1200
#define K 800
#define N 1600
#define TILE_SIZE 32

void mmult_v0(int m, int k, int n, int *a_ptr, int *b_ptr, int *c_ptr)
{
  int i, j, z;
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
    {
      int acc = 0;
      for (z=0; z<k; z++)
        acc += (a_ptr[i*k+z] * b_ptr[z*n+j]);
      c_ptr[i*n+j] = acc;
    }
  }
}

void mmult_v1(int m, int k, int n, int *a_ptr, int *b_ptr, int *c_ptr)
{
  int i, j, z;
  #pragma omp parallel for private (i, j, z)
  for (i=0; i<m; i+=TILE_SIZE)
  {
    for (j=0; j<n; j+=TILE_SIZE)
    {
      int ii, jj;

      int iiMax = (i + TILE_SIZE);
      if (iiMax > M)
        iiMax = M;

      int jjMax = (j + TILE_SIZE);
      if (jjMax > N)
        jjMax = N;

      for (ii=i; ii<iiMax; ii++)
      {
        for (jj=j; jj<jjMax; jj++)
        {
          int acc = 0;
          for (z=0; z<k; z++)
            acc += (a_ptr[ii*k+z] * b_ptr[z*n+jj]);
          c_ptr[ii*n+jj] = acc;
        }
      }
    }
  }
}

void mmult_v2(int m, int k, int n, int *a_ptr, int *b_ptr, int *c_ptr)
{
  int i, j, z;
  #pragma omp parallel for private (i, j, z)
  for (i=0; i<m; i+=TILE_SIZE)
  {
    for (j=0; j<n; j+=TILE_SIZE)
    {
      int ii, jj;

      int iiMax = (i + TILE_SIZE);
      if (iiMax > M)
        iiMax = M;

      int jjMax = (j + TILE_SIZE);
      if (jjMax > N)
        jjMax = N;

      for (ii=i; ii<iiMax; ii++)
      {
        for (jj=j; jj<jjMax; jj++)
        {
          c_ptr[ii*n+jj] = 0;
        }
      }

      for (z=0; z<k; z++)
      {
        for (ii=i; ii<iiMax; ii++)
        {
          for (jj=j; jj<jjMax; jj++)
          {
            c_ptr[ii*n+jj] += (a_ptr[ii*k+z] * b_ptr[z*n+jj]);
          }
        }
      }
    }
  }
}

void mmult_v3(int m, int k, int n, int *a_ptr, int *b_ptr, int *c_ptr)
{
  int i, j, z;
  #pragma omp parallel for private (i, j, z)
  for (i=0; i<m; i+=TILE_SIZE)
  {
    for (j=0; j<n; j+=TILE_SIZE)
    {
      int ii, jj;

      int iiMax = (i + TILE_SIZE);
      if (iiMax > M)
        iiMax = M;

      int jjMax = (j + TILE_SIZE);
      if (jjMax > N)
        jjMax = N;

      for (ii=i; ii<iiMax; ii++)
      {
        for (jj=j; jj<jjMax; jj++)
        {
          c_ptr[ii*n+jj] = 0;
        }
      }

      for (z=0; z<k; z+=TILE_SIZE)
      {
        for (ii=i; ii<iiMax; ii++)
        {
          for (jj=j; jj<jjMax; jj++)
          {
            int zz;

            int zzMax = (z + TILE_SIZE);
            if (zzMax > K)
              zzMax = K;

            for (zz=z; zz<zzMax; zz++)
            {
              c_ptr[ii*n+jj] += (a_ptr[ii*k+zz] * b_ptr[zz*n+jj]);
            }
          }
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  int *a = (int *)calloc(M*K, sizeof(int));
  int *b = (int *)calloc(K*N, sizeof(int));
  int *c = (int *)calloc(M*N, sizeof(int));
  int *r = (int *)calloc(M*N, sizeof(int));
  int i,j,k;

  if (argc < 2) {
    fprintf(stderr, "Missing function index: 1-3\n");
    return 1;
  }
  int idx = atoi(argv[1]);

  srand(time(NULL));

  for (i=0; i<M; i++)
    for(j=0; j<K; j++)
      a[i*K+j] = rand();

  for (i=0; i<K; i++)
    for(j=0; j<N; j++)
      b[i*N+j] = rand();

  ProfilerStart("mmult.prof");

  mmult_v0(M, K, N, (int *)a, (int *)b, (int *)c);

  switch (idx)
  {
    case 1:
      mmult_v1(M, K, N, (int *)a, (int *)b, (int *)r);
      break;
    case 2:
      mmult_v2(M, K, N, (int *)a, (int *)b, (int *)r);
      break;
    case 3:
      mmult_v3(M, K, N, (int *)a, (int *)b, (int *)r);
      break;
    default:
      fprintf(stderr, "Invalid index: %d\n", idx);
      exit(1);
  }

  ProfilerFlush();
  ProfilerStop();

  printf("Verification... ");
  for (i=0; i<M; i++)
    for(j=0; j<N; j++)
      assert(c[i*N+j] == r[i*N+j]);
  printf("done!\n");

  return 0;
}

