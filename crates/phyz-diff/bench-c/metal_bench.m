// Batched throughput of an emitted tang-expr MSL kernel on the local GPU.
//
//   clang -O2 -fobjc-arc -framework Metal -framework Foundation metal_bench.m -o metal_bench
//   ./metal_bench DIR NAME N_INPUTS N_OUT N_CHECK [N_ENVS...]
//
// Reads DIR/NAME.metal, DIR/NAME_inputs.bin and DIR/NAME_reference.bin (as
// written by examples/forward_bench.rs or contact_bench.rs), tiles the
// reference states up to each N_ENVS, and reports error on the first
// copy (first N_CHECK outputs, as bench.c) and GPU time per env-step.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double *load(const char *dir, const char *name, const char *suffix, size_t *n) {
    char p[1024];
    snprintf(p, sizeof p, "%s/%s%s", dir, name, suffix);
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

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 6) {
            fprintf(stderr, "usage: metal_bench DIR NAME N_INPUTS N_OUT N_CHECK [N_ENVS...]\n");
            return 1;
        }
        const char *dir = argv[1], *name = argv[2];
        int ni = atoi(argv[3]), no = atoi(argv[4]), nc = atoi(argv[5]);
        size_t nin, nref;
        double *in64 = load(dir, name, "_inputs.bin", &nin);
        double *ref = load(dir, name, "_reference.bin", &nref);
        int nstates = (int)(nin / ni);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        char path[1024];
        snprintf(path, sizeof path, "%s/%s.metal", dir, name);
        NSString *src = [NSString stringWithContentsOfFile:@(path) encoding:NSUTF8StringEncoding error:nil];
        if (!src) { fprintf(stderr, "cannot read %s\n", path); return 1; }
        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe;
        NSDate *t0 = [NSDate date];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k0"] error:&err];
        if (!pso) { fprintf(stderr, "pipeline: %s\n", err.localizedDescription.UTF8String); return 1; }
        printf("%s on %s: compile %.1f s, max threads/group %lu, exec width %lu\n", name,
               dev.name.UTF8String, -[t0 timeIntervalSinceNow],
               (unsigned long)pso.maxTotalThreadsPerThreadgroup,
               (unsigned long)pso.threadExecutionWidth);
        id<MTLCommandQueue> q = [dev newCommandQueue];

        int sizes[16] = {1024, 4096, 16384, 65536};
        int nsizes = 4;
        if (argc > 6) {
            nsizes = 0;
            for (int i = 6; i < argc && nsizes < 16; i++) sizes[nsizes++] = atoi(argv[i]);
        }
        for (int si = 0; si < nsizes; si++) {
            int n = sizes[si];
            id<MTLBuffer> bin = [dev newBufferWithLength:(size_t)n * ni * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bout = [dev newBufferWithLength:(size_t)n * no * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
            float *pin = bin.contents;
            for (int e = 0; e < n; e++)
                for (int k = 0; k < ni; k++) pin[(size_t)e * ni + k] = (float)in64[(size_t)(e % nstates) * ni + k];
            uint32_t count = n;
            NSUInteger tg = pso.maxTotalThreadsPerThreadgroup < 64 ? pso.maxTotalThreadsPerThreadgroup : 64;
            double best = 1e30;
            for (int rep = 0; rep < 6; rep++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bin offset:0 atIndex:0];
                [enc setBuffer:bout offset:0 atIndex:1];
                [enc setBytes:&count length:sizeof count atIndex:2];
                [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) { fprintf(stderr, "gpu: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
                double t = cb.GPUEndTime - cb.GPUStartTime;
                if (rep > 0 && t < best) best = t;
            }
            float *pout = bout.contents;
            int m = n < nstates ? n : nstates;
            double worst = 0, sum = 0;
            int bad = 0;
            for (int s = 0; s < m; s++) {
                double scale = 1, w = 0;
                for (int k = 0; k < nc; k++) scale = fmax(scale, fabs(ref[(size_t)s * no + k]));
                for (int k = 0; k < nc; k++) {
                    double d = fabs(pout[(size_t)s * no + k] - ref[(size_t)s * no + k]) / scale;
                    if (isnan(d)) { bad++; d = 0; }
                    w = fmax(w, d);
                }
                worst = fmax(worst, w);
                sum += w;
            }
            printf("  envs %6d: %.3f ms/launch, %.4f us/env-step, %.2f M env-steps/s | err worst %.2e mean %.2e%s\n",
                   n, best * 1e3, best * 1e6 / n, n / best / 1e6, worst, sum / m,
                   bad ? " (NaN!)" : "");
        }
    }
    return 0;
}
