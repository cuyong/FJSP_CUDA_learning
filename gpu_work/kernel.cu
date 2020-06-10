#define _CRT_SECURE_NO_WARNINGS

#define MaxGen 200
#define sizeT 100
#define Cross 400

#define rowA 50
#define colA 50
#define Flexibility 10
#define MaxOper 509
#define MaxPopul 1509
#define MaxGoods 50
#define MaxMachine 51

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include<stdio.h>
#include<time.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include <stdio.h>
#include<iostream>
#include<cstring>
#include<string>
#include<cstdio>
#include<algorithm>
#include<stack>
#include<iomanip>
#include<fstream>
#include<curand_kernel.h>

using namespace std;

typedef struct ob {
	int count;
	int t[Flexibility];
	int m[Flexibility];
}ob;

typedef struct good {
	int ans;
	double a[MaxOper];
	int count[MaxOper];
}good;

struct put {
	int start;
	int num;
	int count;
	int end;
}print[MaxOper][MaxOper];

int best[MaxPopul];
int num[MaxGoods];
double *cmper = NULL;
good s[MaxPopul], tmp[MaxPopul];

void unit_c(double *x, int tot)
{
	double sum = 0;
	double multi = 100;
	int i;
	for (i = 1; i <= tot; i++)
	{
		sum += x[i] * x[i];
	}
	sum = sqrt(sum);
	for (i = 1; i <= tot; i++)
	{
		x[i] = x[i] * multi / sum;
	}
}

__device__ void unit(double *x, int tot)
{
	double sum = 0;
	double multi = 100;
	int i;
	for (i = 1; i <= tot; i++)
	{
		sum += x[i] * x[i];
	}
	sum = sqrt(sum);
	for (i = 1; i <= tot; i++)
	{
		x[i] = x[i] * multi / sum;
	}
}

__device__ void unit2(double * x, int * tot)
{
	double sum = 0;
	double multi = 70;
	int i;
	for (i = 1; i <= *tot; i++)
	{
		sum += x[i] * x[i];
	}
	sum = sqrt(sum);
	for (i = 1; i <= *tot; i++)
	{
		x[i] = x[i] * multi / sum;
	}
}

int cmp1(const void * a, const void * b)
{
	return *(cmper + *(int*)a) - *(cmper + *(int*)b);		//�Ƚ�countʱ����ָ��cmper��ƫ��a����С�Ƚ�
}

int cmp2(const void * a, const void *b)
{
	return s[*(int*)a].ans - s[*(int*)b].ans;
}

int cmp3(const void * a, const void *b)
{
	return *(int*)a - *(int*)b;
}

//�����������ͣ��������������ʱ��
__global__ void Calculate(ob **a, good *s, int *facx, int *facy, int *tot, int n) {
	int x = 0;
	if (n == 101) {
		x = blockDim.x * blockIdx.x + threadIdx.x + 1;
	}
	else {
		x = blockDim.x * blockIdx.x + threadIdx.x + 101;
		n += 100;
	}

	int Max = 0;									//Max���ʱ��

	if (x < n) {
		int sx[MaxOper];
		int sy[MaxOper];
		int mt[MaxOper];
		int gt[MaxOper];
		memset(sx, 0, sizeof(sx));
		memset(sy, 0, sizeof(sy));
		memset(mt, 0, sizeof(mt));
		memset(gt, 0, sizeof(gt));

		for (int i = 1; i <= *tot; i++)				//��ѭ����������������
		{
			sx[s[x].count[i]] = facx[i];			//�ڼ��������Ӧ���ڵڼ������
			sy[s[x].count[i]] = facy[i];			//��������Ӧ����ĵڼ�������
		}
		for (int i = 1; i <= *tot; i++)
		{
			int tmp = 0;
			int chosen;
			int time = 100000;
			int chosen_time;
			int len = a[sx[i]][sy[i]].count;

			for (int j = 0; j < len; j++)
			{
				if (mt[a[sx[i]][sy[i]].m[j]] < time)
				{
					time = mt[a[sx[i]][sy[i]].m[j]];
					chosen = a[sx[i]][sy[i]].m[j];
					chosen_time = j;
				}
			}
			tmp = max(mt[chosen], gt[sx[i]]);
			tmp += a[sx[i]][sy[i]].t[chosen_time];
			Max = max(tmp, Max);
			mt[chosen] = tmp;
			gt[sx[i]] = tmp;
		}
		s[x].ans = Max;
	}
}

__device__ void bubbleSort_1(int *base, int tot, double *cmp) {
	int tmp = 0;
	for (int i = 0; i < tot - 1; i++) {
		for (int j = i; j < tot - 1; j++) {
			if (cmp[base[j]] > cmp[base[j + 1]]) {
				tmp = base[j];
				base[j] = base[j + 1];
				base[j + 1] = tmp;
			}
		}
	}
}

__device__ void bubbleSort_2(int *base, int tot) {
	int tmp = 0;
	for (int i = 0; i < tot - 1; i++) {
		for (int j = i; j < tot - 1; j++) {
			if (base[j] > base[j + 1]) {
				tmp = base[j];
				base[j] = base[j + 1];
				base[j + 1] = tmp;
			}
		}
	}
}

__device__ void bubbleSort_3(int *base, int tot, good *s) {
	int tmp = 0;
	for (int i = 0; i < tot - 1; i++) {
		for (int j = i; j < tot - 1; j++) {
			if (s[base[j]].ans > s[base[j + 1]].ans) {
				tmp = base[j];
				base[j] = base[j + 1];
				base[j + 1] = tmp;
			}
		}
	}
}

//����
__global__ void cross(unsigned int *d_rand, int *n, int *num, good *s, int *tot, int critical) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (i < critical)
	{
		double cmp[MaxOper];
		int x = d_rand[i] % sizeT + 1, y = d_rand[2 * i] % sizeT + 1;
		int cur = i + sizeT;
		s[cur].ans = 0;
		for (int j = 1; j <= *tot; j++)
		{
			s[i + sizeT].a[j] = s[x].a[j] - s[y].a[j];
			s[i + sizeT].count[j] = j;
		}
		unit(s[i + sizeT].a, *tot);

		for (int i = 1; i < *tot; i++) {
			cmp[i] = s[cur].a[i];
		}

		bubbleSort_1(s[cur].count + 1, *tot, cmp);
		for (int j = 1; j <= *n; j++)
		{
			bubbleSort_2(s[cur].count + num[j - 1] + 1, num[j] - num[j - 1]);
		}
	}
}

//����
__global__ void variation(unsigned int *d_rand, int *n, int *num, good *s, int *tot, int critical) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (i < critical)
	{
		double del[MaxOper];
		double cmp[MaxOper];
		memset(del, 0, sizeof(double) * 500);
		int cur = i + sizeT + Cross;

		for (int j = 1; j <= *tot; j++)
		{
			del[j] = (d_rand[i] / 2) * (d_rand[2 * i] % 3);
		}

		unit2(del, tot);
		for (int j = 1; j <= *tot; j++)
		{
			s[i + sizeT + Cross].a[j] = s[i].a[j] + del[j];
			s[i + sizeT + Cross].count[j] = j;
		}
		for (int i = 1; i < *tot; i++) {
			cmp[i] = s[cur].a[i];
		}
		s[cur].ans = 0;

		bubbleSort_1(s[cur].count + 1, *tot, cmp);
		for (int j = 1; j <= *n; j++) {
			bubbleSort_2(s[cur].count + num[j - 1] + 1, num[j] - num[j - 1]);
		}
	}
}

//�Ŵ�
__global__ void genetic_1(good *tmp, good *s, int *best, int critical) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (i < critical)
		tmp[i] = s[best[i]];
}
__global__ void genetic_2(good *s, good *tmp, int *tot, int critical) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (i < critical) {
		s[i] = tmp[i];
		unit(s[i].a, *tot);
	}
}

//�Ŵ��㷨
void geneticAl(good *d_s, int *d_num, int *num, int *d_n, int *n, int *d_tot, int *tot, ob **d_a, ob **a,
	int *d_facx, int *facx, int *d_facy, int *facy, good *d_tmp, good *tmp, int *d_best, int *best) {
	*tot = num[*n];										//tot��¼�����ܼ�
	for (int i = 1; i <= sizeT; i++)					//ѭ��sizeT=100�Σ���ʼ��sizeT������
	{
		for (int j = 1; j <= *tot; j++)					//��ѭ����ÿ�������a��50�����������count�еĵ�[j]��ʼ��Ϊj
		{
			s[i].a[j] = rand();
			s[i].count[j] = j;
		}
		unit_c(s[i].a, *tot);							//s[i].a�Ĺ淶��
		cmper = s[i].a;

		qsort(s[i].count + 1, *tot, sizeof(int), cmp1); //����a��С��count����
		for (int j = 1; j <= *n; j++)					//����������Ĺ�������
		{
			qsort(s[i].count + num[j - 1] + 1, num[j] - num[j - 1], sizeof(int), cmp3);
		}
	}

	cudaMemcpy(d_s, s, MaxPopul * sizeof(good), cudaMemcpyHostToDevice);
	int critical = sizeT + 1;
	// ִ��kernel
	Calculate << < 1, 128 >> > (d_a, d_s, d_facx, d_facy, d_tot, critical);//�����������Ⱥ������״ans

	cudaError_t error = cudaGetLastError();
	printf("CUDA error: %s--------------------------------------------------\n", cudaGetErrorString(error));

	unsigned int *d_rand;
	cudaMalloc((void **)&d_rand, 1024 * sizeof(unsigned int));
	curandGenerator_t gen; //�������������
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);	//����1��ָ���㷨
	curandSetPseudoRandomGeneratorSeed(gen, time(NULL));	//����2���������ʼ��

	int counter = 0;
	while (counter <= MaxGen) {//cpu��ִ��ѭ��������kernel
		cout << "counter = " << counter << endl;

		//����
		curandGenerate(gen, d_rand, 1024);					//�����������������
		critical = Cross + 1;
		cross << < 1, 512 >> > (d_rand, d_n, d_num, d_s, d_tot, critical);
		cudaDeviceSynchronize();//�ȴ�ͬ��

		//����
		curandGenerate(gen, d_rand, 1024);
		critical = sizeT + Cross + 1;
		variation << < 1, 512 >> > (d_rand, d_n, d_num, d_s, d_tot, critical);
		cudaDeviceSynchronize();

		//������״ans
		critical = sizeT + Cross * 2 + 1;
		Calculate << < 2, 512 >> > (d_a, d_s, d_facx, d_facy, d_tot, critical);
		cudaDeviceSynchronize();

		//ѡ���Ŵ�
		for (int i = 1; i <= (sizeT + Cross) * 2; i++) {
			best[i] = i;
		}
		cudaMemcpy(s, d_s, MaxPopul * sizeof(good), cudaMemcpyDeviceToHost);
		qsort(best + 1, (sizeT + Cross) * 2, sizeof(int), cmp2);
		cudaMemcpy(d_best, best, MaxPopul * sizeof(int), cudaMemcpyHostToDevice);
		critical = sizeT + Cross + 1;
		genetic_1 << < 1, 128 >> > (d_tmp, d_s, d_best, critical);
		cudaDeviceSynchronize();
		genetic_2 << < 1, 128 >> > (d_s, d_tmp, d_tot, critical);
		cudaDeviceSynchronize();

		counter++;
	}
	curandDestroyGenerator(gen); //�ͷ�ָ��
	cudaFree(d_rand); //�ͷ�GPU���ڴ�ռ�
}

//������
void outputs(int tot, int *facx, int *facy, ob **a, int m)
{
	FILE *fp = fopen("output.txt", "w");
	int x = 1;
	int mt[MaxOper];
	int gt[MaxOper];
	int b[MaxOper];
	int sx[MaxOper];
	int sy[MaxOper];
	memset(mt, 0, sizeof(mt));
	memset(gt, 0, sizeof(gt));
	memset(b, 0, sizeof(b));
	memset(sx, 0, sizeof(sx));
	memset(sy, 0, sizeof(sy));
	int Max = 0;
	int i, j;
	for (i = 1; i <= tot; i++)
	{
		sx[s[x].count[i]] = facx[i];
		sy[s[x].count[i]] = facy[i];
	}
	for (i = 1; i <= tot; i++)
	{
		int tmp;
		int chosen;
		int chosen_time;
		int time = 100000;
		int len = a[sx[i]][sy[i]].count;
		for (j = 0; j < len; j++)
		{
			if (mt[a[sx[i]][sy[i]].m[j]] < time)
			{
				time = mt[a[sx[i]][sy[i]].m[j]];
				chosen = a[sx[i]][sy[i]].m[j];
				chosen_time = j;
			}
		}
		tmp = max(mt[chosen], gt[sx[i]]);
		b[chosen]++;
		print[chosen][b[chosen]].start = tmp;
		print[chosen][b[chosen]].num = sx[i];
		print[chosen][b[chosen]].count = sy[i];
		print[chosen][b[chosen]].end = tmp + a[sx[i]][sy[i]].t[chosen_time];
		tmp += a[sx[i]][sy[i]].t[chosen_time];
		Max = max(tmp, Max);
		mt[chosen] = tmp;
		gt[sx[i]] = tmp;
	}
	for (i = 1; i <= m; i++) {
		fprintf(fp, "M%d", i);
		for (j = 1; j <= b[i]; j++) {
			fprintf(fp, " (%d,%d-%d,%d)", print[i][j].start, print[i][j].num, print[i][j].count, print[i][j].end);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "End %d\n", Max);
	fclose(fp);
}

int main(void)
{
	clock_t s_time, e_time;
	s_time = clock();

	int critical = 0; //���м�����ٽ�ֵ

	//����ָ��
	int *tot, *d_tot, *n, *d_n, *m, *d_m;
	ob **a;
	ob *dataA;
	ob **d_a;
	ob *dataD_a;
	good *d_s;
	good *d_tmp;
	int *facx, *facy;
	int *d_facx, *d_facy;
	int *d_num;
	int *d_best;

	//CPU�����ڴ�
	tot = (int*)malloc(sizeof(int));
	n = (int*)malloc(sizeof(int));
	m = (int*)malloc(sizeof(int));
	a = (ob **)malloc(sizeof(ob*) * rowA);
	dataA = (ob *)malloc(sizeof(ob) * rowA * colA);
	facx = (int*)malloc(sizeof(int) * MaxOper);
	facy = (int*)malloc(sizeof(int) * MaxOper);

	for (int i = 0; i < rowA; i++) {	//����һ��ָ������Ͷ���ָ��������ϵ
		a[i] = dataA + colA * i;
	}

	//GPU�����ڴ�
	cudaMalloc((void **)&d_tot, sizeof(int));
	cudaMalloc((void **)&d_n, sizeof(int));
	cudaMalloc((void **)&d_a, sizeof(ob*) * rowA);
	cudaMalloc((void **)&dataD_a, sizeof(ob) * rowA * colA);
	cudaMalloc((void **)&d_s, sizeof(good) * MaxPopul);
	cudaMalloc((void **)&d_facx, sizeof(int) * MaxOper);
	cudaMalloc((void **)&d_facy, sizeof(int) * MaxOper);
	cudaMalloc((void **)&d_num, sizeof(int) * MaxGoods);
	cudaMalloc((void **)&d_tmp, sizeof(good) * MaxPopul);
	cudaMalloc((void **)&d_best, sizeof(int) * MaxPopul);

	//��ʼ����������
	freopen("Mk01.txt", "r", stdin);
	srand(time(NULL) + 990227);
	*tot = 0;
	memset(s, 0, sizeof(s));
	memset(best, 0, sizeof(best));
	memset(num, 0, sizeof(num));
	int i, j;
	int k = 1;
	char  ch;
	int flag = 0;
	i = 1;
	j = 1;

	cin >> *n >> *m;							//*n����Ʒ������*m�ǻ�������
	char ch1;
	int temp = 0;
	getchar();
	while (i <= *n)
	{
		cin >> temp;
		getchar();
		for (j = 1; j <= temp; j++)
		{
			int tmp1;
			cin >> tmp1;
			getchar();
			a[i - 1][j].count = tmp1;
			for (k = 0; k < tmp1; k++)
			{
				cin >> a[i - 1][j].m[k] >> a[i - 1][j].t[k];
				int x, y;
				x = a[i - 1][j].m[k];
				y = a[i - 1][j].t[k];
			}
			for (int l = 0; l < k; l++)
			{
				for (int kl = 0; kl < k - l - 1; kl++) {
					if (a[i - 1][j].t[kl] > a[i - 1][j].t[kl + 1])
					{
						int tmp = a[i - 1][j].t[kl];
						a[i - 1][j].t[kl] = a[i - 1][j].t[kl + 1];
						a[i - 1][j].t[kl + 1] = tmp;
						tmp = a[i - 1][j].m[kl];
						a[i - 1][j].m[kl] = a[i - 1][j].m[kl + 1];
						a[i - 1][j].m[kl + 1] = tmp;

					}
				}
			}
			facx[++*tot] = i - 1;
			facy[*tot] = j;
		}
		num[i] = temp;
		num[i] += num[i - 1];				//num[i]�洢��i����Ʒ��ʼ�Ĺ�����num[i]
		i++;
	}

	for (int i = 0; i < rowA; i++) {		//����GPU��ַһ��ָ������Ͷ���ָ��������ϵ
		a[i] = dataD_a + colA * i;
	}

	//CPU�����ݿ�����GPU��
	cudaMemcpy(d_tot, tot, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, a, sizeof(ob*) * rowA, cudaMemcpyHostToDevice);
	cudaMemcpy(dataD_a, dataA, sizeof(ob) * rowA * colA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_facx, facx, sizeof(int) * MaxOper, cudaMemcpyHostToDevice);
	cudaMemcpy(d_facy, facy, sizeof(int) * MaxOper, cudaMemcpyHostToDevice);
	cudaMemcpy(d_num, num, sizeof(int) * MaxGoods, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tmp, tmp, sizeof(good) * MaxPopul, cudaMemcpyHostToDevice);
	cudaMemcpy(d_best, best, sizeof(int) * MaxPopul, cudaMemcpyHostToDevice);

	//GeneticAlgorithm
	geneticAl(d_s, d_num, num, d_n, n, d_tot, tot, d_a, a,
		d_facx, facx, d_facy, facy, d_tmp, tmp, d_best, best);

	for (int i = 0; i < rowA; i++) {	//����һ��ָ������Ͷ���ָ��������ϵ
		a[i] = dataA + colA * i;
	}
	cudaMemcpy(s, d_s, MaxPopul * sizeof(good), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 100; i++) {
		cout << s[i + 1].ans << "\t";
	}
	outputs(*tot, facx, facy, a, *m);

	cudaFree(d_tot);
	cudaFree(d_n);
	cudaFree(d_a);
	cudaFree(dataD_a);
	cudaFree(d_facx);
	cudaFree(d_facy);
	cudaFree(d_num);
	cudaFree(d_tmp);
	cudaFree(d_best);
	cudaFree(d_s);

	e_time = clock();
	cout << "Total time:" << (double)(e_time - s_time) / CLOCKS_PER_SEC << "S" << endl;

	return 0;
}