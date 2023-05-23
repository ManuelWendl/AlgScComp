# include <immintrin.h>
# include <stdio.h>
# include <stdbool.h>
# include <math.h>

# define pi 3.14159265358979323846264338327950288
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

struct ComplexArray *butterfly_vec(struct ComplexArray X, bool inverse, int N){
    printf("vec");
    float theta;

    __m128 wr, wi, wzr, wzi, Tempr, Tempi;

    float b;
    if (inverse){
        b = 1;
    }else{
        b = -1;
    }
    for (int L = 2; L<= N; L*=2) {
        theta = b*2*pi/L;
        bool greater;
        if (L/2>SIMD_LENGTH){
            greater = true;
        } else{
            greater = false;
        }
        for (int k = 0; k < N; k += L) {
            if (greater){
            for (int j = 0; j < L / 2; j+= SIMD_LENGTH) {
                wr = _mm_set_ps(cosf(theta * (float) j), cosf(theta * (float) (j + 1)),
                                   cosf(theta * (float) (j + 2)), cosf(theta * (float) (j + 3)));
                wi = _mm_set_ps(sinf(theta * (float) j), sinf(theta * (float) (j + 1)),
                                   sinf(theta * (float) (j + 2)), sinf(theta * (float) (j + 3)));

                wzr = _mm_sub_ps(_mm_mul_ps(wr, _mm_load_ps((float *) X.real + k + j + L / 2)), _mm_mul_ps(wi, _mm_load_ps((float *) X.imag + k + j + L / 2)));
                wzi = _mm_sub_ps(_mm_mul_ps(wr, _mm_load_ps((float *) X.imag + k + j + L / 2)), _mm_mul_ps(wi, _mm_load_ps((float *) X.real + k + j + L / 2)));

                _mm_store_ps((float *) X.real + k + j + L / 2, _mm_sub_ps(_mm_load_ps((float *) X.real + k + j), wzr));
                _mm_store_ps((float *) X.imag + k + j + L / 2, _mm_sub_ps(_mm_load_ps((float *) X.imag + k + j), wzi));

                Tempr = _mm_add_ps(_mm_load_ps((float *) X.real + k + j), wzr);
                Tempi = _mm_add_ps(_mm_load_ps((float *) X.imag + k + j), wzi);

                _mm_store_ps((float *) X.real + k + j, Tempr);
                _mm_store_ps((float *) X.imag + k + j, Tempi);
            }
            }else{
                for (int j = 0; j < L / 2; j++) {
                    float wr2 = cosf(theta * (float) j);
                    float wi2 = sinf(theta * (float) j);
                    float wzr2 = wr2 * X.real[k + j + L / 2] - wi2 * X.imag[k + j + L / 2];
                    float wzi2 = wr2 * X.imag[k + j + L / 2] + wi2 * X.real[k + j + L / 2];
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
    printf("v1");
    float theta;
    struct Complex w,wz;

    int b;
    if (inverse){
        b = 1;
    }else{
        b = -1;
    }

    for (int L = 2; L<= N; L*=2) {
        theta = b*2*pi/L;
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

int main(){
    float a[] = {1,5,3,7,2,6,4,8};
    float b[] = {8,4,6,2,7,3,5,1};
    int N = sizeof(a)/sizeof(a[0]);

    struct ComplexArray x;
    x.real = a;
    x.imag = b;

    printcomplex(x,N);

    struct ComplexArray *z = butterfly_vec(x,false,N);

    printcomplex(*z,N);

    freeMemory(z);

    float c[] = {1,5,3,7,2,6,4,8};
    float d[] = {8,4,6,2,7,3,5,1};

    struct ComplexArray y;
    y.real = c;
    y.imag = d;

    printcomplex(y,N);

    struct ComplexArray *l = butterfly_v1(y,false,N);

    printcomplex(*l,N);


    freeMemory(l);
    return 0;
}