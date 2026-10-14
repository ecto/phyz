// The split contact pipeline (k_kin, k_solve, k_finish) batched on the GPU.
//
//   clang -O2 -fobjc-arc -framework Metal -framework Foundation metal_split.m -o metal_split
//   ./metal_split OUT NX NREC NSCR NFIN DIM NOUT SWEEPS [N_ENVS...]
//
// Reads OUT/split.metal and the contact_* inputs/reference written by
// examples/contact_bench.rs; tiles the 1000 reference states up to N_ENVS.
// Every GPU buffer is structure-of-arrays (`slot * n + env`). With -DWITH_C
// and OUT/split.c linked in, also diffs each stage against the C pipeline.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef WITH_C
void k_kin(const float *inputs, float *outputs, int count);
void k_solve(const float *rec, const float *warm, const float *x, float *scratch, float *f_out,
             float *fin, int count, int sweeps);
void k_finish(const float *inputs, float *outputs, int count);

static void diff(const char *what, const float *g, const float *c, size_t n, int stride) {
    double worst = 0;
    size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)g[i] - c[i]) / fmax(1.0, fabs(c[i]));
        if (d > worst || isnan(d)) { worst = isnan(d) ? INFINITY : d; at = i; }
    }
    printf("  %-8s worst rel diff %.2e at env %zu slot %zu (gpu %g, c %g)\n", what, worst, at / stride, at % stride,
           g[at], c[at]);
}
#endif

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

/// Structure-of-arrays (`slot * n + env`) to array-of-structs, freshly allocated.
static float *aos(const float *soa, int n, int width) {
    float *out = malloc(sizeof(float) * (size_t)n * width);
    for (int e = 0; e < n; e++)
        for (int k = 0; k < width; k++) out[(size_t)e * width + k] = soa[(size_t)k * n + e];
    return out;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    id<MTLComputePipelineState> p =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
    if (!p) { fprintf(stderr, "pipeline %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String); exit(1); }
    return p;
}

static id<MTLBuffer> buf(id<MTLDevice> dev, size_t floats) {
    return [dev newBufferWithLength:floats * sizeof(float) options:MTLResourceStorageModeShared];
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 9) {
            fprintf(stderr, "usage: metal_split DIR NX NREC NSCR NFIN DIM NOUT SWEEPS [N_ENVS...]\n");
            return 1;
        }
        const char *dir = argv[1];
        int nx = atoi(argv[2]), nrec = atoi(argv[3]), nscr = atoi(argv[4]), nfin = atoi(argv[5]);
        int dim = atoi(argv[6]), nout = atoi(argv[7]);
        uint32_t sweeps = atoi(argv[8]);
        size_t nin, nref;
        double *in64 = load(dir, "contact_inputs.bin", &nin);
        double *ref = load(dir, "contact_reference.bin", &nref);
        int stride_in = nx + dim, stride_ref = nout + dim;
        int nstates = (int)(nin / stride_in);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        char path[1024];
        snprintf(path, sizeof path, "%s/split.metal", dir);
        NSString *src = [NSString stringWithContentsOfFile:@(path) encoding:NSUTF8StringEncoding error:nil];
        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe;
        NSDate *t0 = [NSDate date];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> kin = make_pipeline(dev, lib, @"k_kin"), sol = make_pipeline(dev, lib, @"k_solve"),
                                    fin = make_pipeline(dev, lib, @"k_finish");
        printf("split on %s, %u sweeps: compile %.1f s\n", dev.name.UTF8String, sweeps, -[t0 timeIntervalSinceNow]);
        id<MTLCommandQueue> q = [dev newCommandQueue];

        int sizes[16] = {4096, 16384, 65536};
        int nsizes = 3;
        if (argc > 9) {
            nsizes = 0;
            for (int i = 9; i < argc && nsizes < 16; i++) sizes[nsizes++] = atoi(argv[i]);
        }
        for (int si = 0; si < nsizes; si++) {
            int n = sizes[si];
            id<MTLBuffer> bx = buf(dev, (size_t)n * nx), bw = buf(dev, (size_t)n * dim),
                          brec = buf(dev, (size_t)n * nrec), bscr = buf(dev, (size_t)n * nscr),
                          bf = buf(dev, (size_t)n * dim), bfin = buf(dev, (size_t)n * nfin),
                          bout = buf(dev, (size_t)n * nout);
            float *px = bx.contents, *pw = bw.contents;
            float *ax = malloc(sizeof(float) * n * nx), *aw = malloc(sizeof(float) * n * dim);
            for (int e = 0; e < n; e++) {
                int s = e % nstates;
                for (int k = 0; k < nx; k++) {
                    ax[(size_t)e * nx + k] = (float)in64[(size_t)s * stride_in + k];
                    px[(size_t)k * n + e] = ax[(size_t)e * nx + k];
                }
                for (int k = 0; k < dim; k++) {
                    aw[(size_t)e * dim + k] = (float)in64[(size_t)s * stride_in + nx + k];
                    pw[(size_t)k * n + e] = aw[(size_t)e * dim + k];
                }
            }
            uint32_t count = n;
            double best = 1e30;
            for (int rep = 0; rep < 6; rep++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                MTLSize grid = MTLSizeMake(n, 1, 1), tg = MTLSizeMake(64, 1, 1);
                [enc setComputePipelineState:kin];
                [enc setBuffer:bx offset:0 atIndex:0];
                [enc setBuffer:brec offset:0 atIndex:1];
                [enc setBytes:&count length:4 atIndex:2];
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc setComputePipelineState:sol];
                [enc setBuffer:brec offset:0 atIndex:0];
                [enc setBuffer:bw offset:0 atIndex:1];
                [enc setBuffer:bx offset:0 atIndex:2];
                [enc setBuffer:bscr offset:0 atIndex:3];
                [enc setBuffer:bf offset:0 atIndex:4];
                [enc setBuffer:bfin offset:0 atIndex:5];
                [enc setBytes:&count length:4 atIndex:6];
                [enc setBytes:&sweeps length:4 atIndex:7];
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc setComputePipelineState:fin];
                [enc setBuffer:bfin offset:0 atIndex:0];
                [enc setBuffer:bout offset:0 atIndex:1];
                [enc setBytes:&count length:4 atIndex:2];
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) { fprintf(stderr, "gpu: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
                double t = cb.GPUEndTime - cb.GPUStartTime;
                if (rep > 0 && t < best) best = t;
            }
            float *po = aos(bout.contents, n, nout);
#ifdef WITH_C
            if (si == 0) {
                float *crec = malloc(sizeof(float) * n * nrec), *cscr = malloc(sizeof(float) * n * nscr);
                float *cf = malloc(sizeof(float) * n * dim), *cfin = malloc(sizeof(float) * n * nfin);
                float *cout = malloc(sizeof(float) * n * nout);
                k_kin(ax, crec, n);
                k_solve(crec, aw, ax, cscr, cf, cfin, n, sweeps);
                k_finish(cfin, cout, n);
                diff("rec", aos(brec.contents, n, nrec), crec, (size_t)n * nrec, nrec);
                diff("f", aos(bf.contents, n, dim), cf, (size_t)n * dim, dim);
                diff("fin", aos(bfin.contents, n, nfin), cfin, (size_t)n * nfin, nfin);
                diff("out", po, cout, (size_t)n * nout, nout);
                float *gf = aos(bfin.contents, n, nfin);
                printf("  per v1 slot worst abs diff:");
                for (int j = nfin - 28; j < nfin; j++) {
                    double w = 0;
                    for (int e = 0; e < n; e++) w = fmax(w, fabs(gf[(size_t)e * nfin + j] - cfin[(size_t)e * nfin + j]));
                    printf(" %.1e", w);
                }
                printf("\n");
            }
#endif
            int m = n < nstates ? n : nstates;
            double worst = 0, sum = 0;
            int bad = 0;
            for (int s = 0; s < m; s++) {
                double scale = 1, w = 0;
                for (int k = 0; k < nout; k++) scale = fmax(scale, fabs(ref[(size_t)s * stride_ref + k]));
                for (int k = 0; k < nout; k++) {
                    double d = fabs(po[(size_t)s * nout + k] - ref[(size_t)s * stride_ref + k]) / scale;
                    if (isnan(d)) { bad++; d = 0; }
                    w = fmax(w, d);
                }
                worst = fmax(worst, w);
                sum += w;
            }
            if (getenv("SPLIT_STAGES")) {
                // Each kernel alone, best of 5, in its own command buffer.
                id<MTLComputePipelineState> ps[3] = {kin, sol, fin};
                const char *names[3] = {"kin", "solve", "finish"};
                for (int st = 0; st < 3; st++) {
                    double b = 1e30;
                    for (int rep = 0; rep < 6; rep++) {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:ps[st]];
                        if (st == 0) {
                            [enc setBuffer:bx offset:0 atIndex:0];
                            [enc setBuffer:brec offset:0 atIndex:1];
                            [enc setBytes:&count length:4 atIndex:2];
                        } else if (st == 1) {
                            [enc setBuffer:brec offset:0 atIndex:0];
                            [enc setBuffer:bw offset:0 atIndex:1];
                            [enc setBuffer:bx offset:0 atIndex:2];
                            [enc setBuffer:bscr offset:0 atIndex:3];
                            [enc setBuffer:bf offset:0 atIndex:4];
                            [enc setBuffer:bfin offset:0 atIndex:5];
                            [enc setBytes:&count length:4 atIndex:6];
                            [enc setBytes:&sweeps length:4 atIndex:7];
                        } else {
                            [enc setBuffer:bfin offset:0 atIndex:0];
                            [enc setBuffer:bout offset:0 atIndex:1];
                            [enc setBytes:&count length:4 atIndex:2];
                        }
                        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        double t = cb.GPUEndTime - cb.GPUStartTime;
                        if (rep > 0 && t < b) b = t;
                    }
                    printf("    %-6s %.3f ms (%.4f us/env)\n", names[st], b * 1e3, b * 1e6 / n);
                }
            }
            printf("  envs %6d: %.3f ms/step, %.4f us/env-step, %.2f M env-steps/s | err worst %.2e mean %.2e%s\n",
                   n, best * 1e3, best * 1e6 / n, n / best / 1e6, worst, sum / m, bad ? " (NaN!)" : "");
        }
    }
    return 0;
}
