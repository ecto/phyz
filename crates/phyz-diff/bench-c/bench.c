// Bench the emitted ABA kernel: batch-1 latency, batched throughput, and
// worst relative error against phyz_rigid::aba.
//
//   cc -O2 -DREAL=float  bench.c OUT/step_f32.c -lm -o step32
//   cc -O2 -DREAL=double bench.c OUT/step_f64.c -lm -o step64
//   ./step32 OUT step N_INPUTS N_OUT   (sizes from OUT/meta.txt)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef REAL
#define REAL float
#endif

void k0(const REAL *inputs, REAL *outputs, int count);

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
    if (argc < 5) { fprintf(stderr, "usage: bench DIR NAME N_INPUTS N_OUT\n"); return 1; }
    const char *dir = argv[1], *name = argv[2];
    int ni = atoi(argv[3]), nv = atoi(argv[4]);
    // Optional: compare only the first n_check outputs (e.g. q', v' and not
    // contact impulses, whose null-space part is not unique).
    int n_check = argc > 5 ? atoi(argv[5]) : nv;
    char fin[256], fref[256];
    snprintf(fin, sizeof fin, "%s_inputs.bin", name);
    snprintf(fref, sizeof fref, "%s_reference.bin", name);
    size_t nin, nref;
    double *in64 = load(dir, fin, &nin);
    double *ref = load(dir, fref, &nref);
    int n = (int)(nin / ni);

    REAL *in = malloc(nin * sizeof(REAL));
    REAL *out = malloc((size_t)n * nv * sizeof(REAL));
    for (size_t i = 0; i < nin; i++) in[i] = (REAL)in64[i];

    // Error against the reference, scaled per state by max(1, |qdd|_inf).
    k0(in, out, n);
    double worst = 0, sum = 0;
    for (int s = 0; s < n; s++) {
        double scale = 1;
        for (int k = 0; k < n_check; k++) scale = fmax(scale, fabs(ref[s * nv + k]));
        double w = 0;
        for (int k = 0; k < n_check; k++) w = fmax(w, fabs(out[s * nv + k] - ref[s * nv + k]) / scale);
        worst = fmax(worst, w);
        sum += w;
    }

    // Batch 1 latency.
    int reps = 5;
    double *lat = malloc((size_t)n * reps * sizeof(double));
    for (int r = 0; r < reps; r++)
        for (int s = 0; s < n; s++) {
            double t = now_ns();
            k0(in + (size_t)s * ni, out + (size_t)s * nv, 1);
            lat[r * n + s] = now_ns() - t;
        }
    qsort(lat, (size_t)n * reps, sizeof(double), cmp);
    size_t m = (size_t)n * reps - 1;

    // Batched throughput (single thread).
    double t = now_ns();
    for (int r = 0; r < reps; r++) k0(in, out, n);
    double per = (now_ns() - t) / ((double)n * reps);

    printf("%s: err worst %.2e mean %.2e | batch1 us median %.2f p99 %.2f p999 %.2f | batched %.2f us/state\n",
           sizeof(REAL) == 4 ? "f32" : "f64", worst, sum / n, lat[m / 2] / 1e3,
           lat[(size_t)(m * 0.99)] / 1e3, lat[(size_t)(m * 0.999)] / 1e3, per / 1e3);
    return 0;
}
