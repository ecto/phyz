// Bench the split contact pipeline (traced kinematics, rolled solve, traced
// finish) on the CPU in f32, against phyz's reference.
//
//   cc -O2 split_bench.c OUT/split.c -lm -o split_bench
//   ./split_bench OUT NX NREC NSCR NFIN DIM NOUT SWEEPS   (sizes: OUT/contact_meta.txt)
//
// Reads OUT/contact_inputs.bin ([x (NX), warm (DIM)] per state) and
// OUT/contact_reference.bin ([q', v' (NOUT), f (DIM)] per state).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void k_kin(const float *inputs, float *outputs, int count);
void k_solve(const float *rec, const float *warm, const float *x, float *scratch, float *f_out,
             float *fin, int count, int sweeps);
void k_finish(const float *inputs, float *outputs, int count);

static double *load(const char *dir, const char *name, size_t *n) {
    char p[1024];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END);
    *n = ftell(f) / sizeof(double);
    fseek(f, 0, SEEK_SET);
    double *x = malloc(*n * sizeof(double));
    if (fread(x, sizeof(double), *n, f) != *n) { perror("read"); exit(1); }
    fclose(f);
    return x;
}

static double now_ns(void) {
#ifdef __APPLE__
    return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#endif
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: split_bench DIR NX NREC NSCR NFIN DIM NOUT SWEEPS\n");
        return 1;
    }
    const char *dir = argv[1];
    int nx = atoi(argv[2]), nrec = atoi(argv[3]), nscr = atoi(argv[4]), nfin = atoi(argv[5]);
    int dim = atoi(argv[6]), nout = atoi(argv[7]), sweeps = atoi(argv[8]);
    size_t nin, nref;
    double *in64 = load(dir, "contact_inputs.bin", &nin);
    double *ref = load(dir, "contact_reference.bin", &nref);
    int stride_in = nx + dim, stride_ref = nout + dim;
    int n = (int)(nin / stride_in);

    float *x = malloc(sizeof(float) * n * nx), *warm = malloc(sizeof(float) * n * dim);
    for (int e = 0; e < n; e++) {
        for (int k = 0; k < nx; k++) x[e * nx + k] = (float)in64[e * stride_in + k];
        for (int k = 0; k < dim; k++) warm[e * dim + k] = (float)in64[e * stride_in + nx + k];
    }
    float *rec = malloc(sizeof(float) * n * nrec), *scr = malloc(sizeof(float) * n * nscr);
    float *f = malloc(sizeof(float) * n * dim), *fin = malloc(sizeof(float) * n * nfin);
    float *out = malloc(sizeof(float) * n * nout);

    k_kin(x, rec, n);
    k_solve(rec, warm, x, scr, f, fin, n, sweeps);
    k_finish(fin, out, n);
    double worst = 0, sum = 0;
    for (int s = 0; s < n; s++) {
        double scale = 1, w = 0;
        for (int k = 0; k < nout; k++) scale = fmax(scale, fabs(ref[(size_t)s * stride_ref + k]));
        for (int k = 0; k < nout; k++)
            w = fmax(w, fabs(out[(size_t)s * nout + k] - ref[(size_t)s * stride_ref + k]) / scale);
        worst = fmax(worst, w);
        sum += w;
    }

    int reps = 5;
    double *lat = malloc(sizeof(double) * n * reps);
    for (int r = 0; r < reps; r++)
        for (int e = 0; e < n; e++) {
            double t = now_ns();
            k_kin(x + (size_t)e * nx, rec + (size_t)e * nrec, 1);
            k_solve(rec + (size_t)e * nrec, warm + (size_t)e * dim, x + (size_t)e * nx,
                    scr + (size_t)e * nscr, f + (size_t)e * dim, fin + (size_t)e * nfin, 1, sweeps);
            k_finish(fin + (size_t)e * nfin, out + (size_t)e * nout, 1);
            lat[r * n + e] = now_ns() - t;
        }
    qsort(lat, (size_t)n * reps, sizeof(double), cmp);
    size_t m = (size_t)n * reps - 1;

    double t = now_ns();
    for (int r = 0; r < reps; r++) {
        k_kin(x, rec, n);
        k_solve(rec, warm, x, scr, f, fin, n, sweeps);
        k_finish(fin, out, n);
    }
    double per = (now_ns() - t) / ((double)n * reps);

    printf("split f32, %d sweeps: err worst %.2e mean %.2e | batch1 us median %.2f p99 %.2f p999 %.2f | batched %.2f us/state\n",
           sweeps, worst, sum / n, lat[m / 2] / 1e3, lat[(size_t)(m * 0.99)] / 1e3,
           lat[(size_t)(m * 0.999)] / 1e3, per / 1e3);
    return 0;
}
