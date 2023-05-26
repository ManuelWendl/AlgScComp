# include <immintrin.h>
# include <stdio.h>
# include <stdbool.h>
# include <math.h>

# define pi 3.1415926535897932384626
# define SIMD_LENGTH 4

struct ComplexArray {
    float *real;
    float *imag;
};

struct Complex {
    float real;
    float imag;
};

void printcomplex(struct ComplexArray X,int N){
    printf("\n z = ");
    for (int i = 0; i<N; i++){
        printf("%f + i%f ;",1.0/N*X.real[i],1.0/N*X.imag[i]);
    }
    printf("\n");
}

void freeMemory(struct ComplexArray *X){
    free(X);
}

void butterfly(float *X, int N){
    for (int n=0; n<N; n++){
        int j = 0; int m = n;
        for (int i = 0; i<log2(N); i++){
            j = 2*j + m%2; m = m/2;
        }
        if (j<n){
            float temp = X[j];
            X[j] = X[n]; X[n] = temp;
        }
    }
}

struct ComplexArray *butterfly_vec(struct ComplexArray X, bool inverse, int N){
    float theta, Wr[N-1], Wi[N-1], wr2, wi2, wzr2, wzi2;
    __m128 wr, wi, wzr, wzi, tempr, tempi;

    butterfly(X.real,N);
    butterfly(X.imag,N);

    float b = inverse ? 1 : -1;

    for (int L = 2; L<= N; L*=2){
        theta = (float) (b*2*pi/L);
        for (int j =0; j<L/2; j++){
            Wr[L/2+j] = cosf(theta * (float) j);
            Wi[L/2+j] = sinf(theta * (float) j);
        }
    }

    for (int L = 2; L<= N; L*=2) {
        for (int k = 0; k < N; k += L) {
            if (L/2 >= SIMD_LENGTH){
                for (int j = 0; j < L / 2; j+= SIMD_LENGTH) {
                    wr = _mm_load_ps((float*) Wr+L/2+j);
                    wi = _mm_load_ps((float*) Wi+L/2+j);

                    wzr = _mm_sub_ps(_mm_mul_ps(wr, _mm_load_ps((float *) X.real + k + j + L / 2)), _mm_mul_ps(wi, _mm_load_ps((float *) X.imag + k + j + L / 2)));
                    wzi = _mm_add_ps(_mm_mul_ps(wr, _mm_load_ps((float *) X.imag + k + j + L / 2)), _mm_mul_ps(wi, _mm_load_ps((float *) X.real + k + j + L / 2)));

                    _mm_store_ps((float *) X.real + k + j + L / 2, _mm_sub_ps(_mm_load_ps((float *) X.real + k + j), wzr));
                    _mm_store_ps((float *) X.imag + k + j + L / 2, _mm_sub_ps(_mm_load_ps((float *) X.imag + k + j), wzi));

                    _mm_store_ps((float *) X.real + k + j, _mm_add_ps(_mm_load_ps((float *) X.real + k + j), wzr));
                    _mm_store_ps((float *) X.imag + k + j, _mm_add_ps(_mm_load_ps((float *) X.imag + k + j), wzi));
                }
            }else {
                for (int j = 0; j < L / 2; j++) {
                    theta = (float) (b*2*pi/L);
                    wr2 = cosf(theta*(float)j);
                    wi2 = sinf(theta*(float)j);
                    wzr2 = wr2 * X.real[k + j + L / 2] - wi2 * X.imag[k + j + L / 2];
                    wzi2 = wr2 * X.imag[k + j + L / 2] + wi2 * X.real[k + j + L / 2];
                    X.real[k + j + L / 2] = X.real[k + j] - wzr2;
                    X.imag[k + j + L / 2] = X.imag[k + j] - wzi2;
                    X.real[k + j] = X.real[k + j] + wzr2;
                    X.imag[k + j] = X.imag[k + j] + wzi2;
                }
            }
        }
    }
    struct ComplexArray *x = malloc(sizeof(struct ComplexArray));
    x->real = X.real;
    x->imag = X.imag;
    return x;
}

struct ComplexArray *butterfly_v1(struct ComplexArray X, bool inverse, int N) {

    butterfly(X.real,N);
    butterfly(X.imag,N);

    float theta;
    struct Complex w,wz;

    float b = inverse ? 1 : -1;

    for (int L = 2; L<= N; L*=2) {
        theta = (float) (b*2*pi/L);
        for (int k = 0; k < N; k += L) {
            for (int j = 0; j < L / 2; j++) {

                w.real = cosf(theta*(float)j);
                w.imag = sinf(theta*(float)j);
                wz.real = w.real * X.real[k+j+L/2] - w.imag * X.imag[k+j+L/2];
                wz.imag = w.real * X.imag[k+j+L/2] + w.imag * X.real[k+j+L/2];
                X.real[k + j + L / 2] = X.real[k + j] - wz.real;
                X.imag[k + j + L / 2] = X.imag[k + j] - wz.imag;
                X.real[k + j] = X.real[k + j] + wz.real;
                X.imag[k + j] = X.imag[k + j] + wz.imag;
            }
        }
    }
    struct ComplexArray *x = malloc(sizeof(struct ComplexArray));
    x->real = X.real;
    x->imag = X.imag;
    return x;
}

struct ComplexArray *butterfly_v2(struct ComplexArray X, bool inverse, int N) {

    butterfly(X.real,N);
    butterfly(X.imag,N);

    float theta;
    struct Complex w,wz;

    float b = inverse ? 1 : -1;

    for (int L = 2; L<= N; L*=2) {
        theta = (float) (b*2*pi/L);
        for (int j = 0; j < L / 2; j++) {
            w.real = cosf(theta*(float)j);
            w.imag = sinf(theta*(float)j);
            for (int k = 0; k < N; k+=L) {
                wz.real = w.real * X.real[k+j+L/2] - w.imag * X.imag[k+j+L/2];
                wz.imag = w.real * X.imag[k+j+L/2] + w.imag * X.real[k+j+L/2];
                X.real[k + j + L / 2] = X.real[k + j] - wz.real;
                X.imag[k + j + L / 2] = X.imag[k + j] - wz.imag;
                X.real[k + j] = X.real[k + j] + wz.real;
                X.imag[k + j] = X.imag[k + j] + wz.imag;
            }
        }
    }
    struct ComplexArray *x = malloc(sizeof(struct ComplexArray));
    x->real = X.real;
    x->imag = X.imag;
    return x;
}

int main(){
    float a[] = {1,2,3,4,5,6,7,8};
    float b[] = {8,7,6,5,4,3,2,1};
    int N = sizeof(a)/sizeof(a[0]);

    struct ComplexArray x;
    x.real = a;
    x.imag = b;

    printcomplex(x,N);

    struct ComplexArray *z = butterfly_vec(x,false,N);

    printcomplex(*z,N);

    freeMemory(z);
    return 0;
}