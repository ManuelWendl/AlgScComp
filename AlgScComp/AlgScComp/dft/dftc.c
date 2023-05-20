# include <immintrin.h>
# include <stdio.h>
# include <math.h>
# include <stdbool.h>
# include <complex.h>
# include <Python/Python.h>

# define pi 3.14159265358979323846264338327950288
# define SIMD_LENGTH 8

void butterfly_vec(float complex *X, bool inverse, int N){
    int b;
    if (inverse){
        b = 1;
    }else{
        b = -1;
    }

    printf("%d",b);

    for (int L = 2; L<= N; L*=2){
        int d;
        if (L/2 < SIMD_LENGTH){
            d = L/2;
        }else{
            d = SIMD_LENGTH;
        }
        for (int k = 0; k<N; k+=L){
            for (int j=0; j<L/2; j+=SIMD_LENGTH){
                int kjStart = k+j;
                int kjEnd = k+j+d-1;

                __m256i w = _mm256_set1_epi64x(w(0,L,b),w(1,L,b),w(2,L,b),w(3,L,b),w(4,L,b),w(5,L,b),w(6,L,b),w(7,L,b))
                float complex wz = cexp(b * 2 * pi * I * j / L) *  X[k + j + L / 2];
                X[k + j + L / 2] = X[k + j] - wz;
                X[k + j] = X[k + j] + wz;
            }
        }
    }
}

complex double w(j,L,b){
    return cexp(b * 2 * pi * I * j / L);
}

void butterfly_v1(float complex *X, bool inverse, int N){
    int b;
    if (inverse){
        b = 1;
    }else{
        b = -1;
    }

    printf("%d",b);

    for (int L = 2; L<= N; L*=2){
        for (int k = 0; k<N; k+=L){
            for (int j=0; j<L/2; j++){
                float complex wz = cexp(b * 2 * pi * I * j / L) *  X[k + j + L / 2];
                X[k + j + L / 2] = X[k + j] - wz;
                X[k + j] = X[k + j] + wz;
            }
        }
    }
}


int main(){
    complex float a[] = {1,2,3,4,5,6,7,8};
    int N = sizeof(a)/sizeof(a[0]);
    butterfly_v1(a,true,N);

    return 0;
}