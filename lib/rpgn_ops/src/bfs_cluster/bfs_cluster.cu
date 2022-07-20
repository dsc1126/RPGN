/*
Ball Query, Guided with Regional Purity
*/

#include "bfs_cluster.h"
#include "../cuda_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


/* ================================== ballquery_batch_p ================================== */
__global__ void ballquery_batch_p_cuda_(int n, int meanActive, float radius, const float *xyz, const int *batch_idxs, const int *batch_offsets, int *idx, int *start_len, int *cumsum) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= n) return;

    start_len += (pt_idx * 2);
    int idx_temp[1000];

    float radius2 = radius * radius;
    float radius2_shifted = radius * radius;
    float o_x = xyz[pt_idx * 11 + 0];
    float o_y = xyz[pt_idx * 11 + 1];
    float o_z = xyz[pt_idx * 11 + 2];
    float o_regional = xyz[pt_idx * 11 + 3];
    float o_x_direction = xyz[pt_idx * 11 + 4];
    float o_y_direction = xyz[pt_idx * 11 + 5];
    float o_z_direction = xyz[pt_idx * 11 + 6];
    float o_x_offset = xyz[pt_idx * 11 + 7];
    float o_y_offset = xyz[pt_idx * 11 + 8];
    float o_z_offset = xyz[pt_idx * 11 + 9];
    float o_size_class = xyz[pt_idx * 11 + 10];

    float o_size = o_size_class * 0.2;
    float o_size2 = o_size * o_size;
    if (o_size2 >= 1){
        o_size2= o_size2 * 10;
    }

    int batch_idx = batch_idxs[pt_idx];
    int start = batch_offsets[batch_idx];
    int end = batch_offsets[batch_idx + 1];

    int cnt = 0;

    for(int k = start; k < end; k++){
        float x = xyz[k * 11 + 0];
        float y = xyz[k * 11 + 1];
        float z = xyz[k * 11 + 2];
        float regional = xyz[k * 11 + 3];
        float x_direction = xyz[k * 11 + 4];
        float y_direction = xyz[k * 11 + 5];
        float z_direction = xyz[k * 11 + 6];
        float x_offset = xyz[k * 11 + 7];
	    float y_offset = xyz[k * 11 + 8];
	    float z_offset = xyz[k * 11 + 9];
	    //float size_class = xyz[k * 11 + 10];

        if(o_regional==1 && regional!=2) {
            radius2 = (radius+0.0) * (radius+0.00);
            radius2_shifted = (radius+0.02) * (radius+0.02);
            float d2 = (o_x - x) * (o_x - x) + (o_y - y) * (o_y - y) + (o_z - z) * (o_z - z);
            float d2_shifted = ((o_x+o_x_offset) - (x+x_offset)) * ((o_x+o_x_offset) - (x+x_offset)) + ((o_y+o_y_offset) - (y+y_offset)) * ((o_y+o_y_offset) - (y+y_offset)) + ((o_z+o_z_offset) - (z+z_offset)) * ((o_z+o_z_offset) - (z+z_offset));
            if(d2_shifted < radius2_shifted) { 
                if(cnt < 1000){
                    idx_temp[cnt] = k;
                }
                else{
                    break;
                }
                ++cnt;
            }
        }
        
        else if(o_regional==0 && regional==2) {
            float cos_theta = (o_x_direction * x_direction) + (o_y_direction * y_direction) + (o_z_direction * z_direction);
            if(cos_theta>0.8) {
                radius2 = (radius+0.15) * (radius+0.15);
            else if(cos_theta<=0.8) {
                radius2 = 0.00001;
            }
            float d2 = (o_x - x) * (o_x - x) + (o_y - y) * (o_y - y) + (o_z - z) * (o_z - z);
            if(d2 < radius2){
                if(cnt < 1000){
                    idx_temp[cnt] = k;
                }
                else{
                    break;
                }
                ++cnt;
            }
        }
    }

    start_len[0] = atomicAdd(cumsum, cnt);
    start_len[1] = cnt;

    int thre = n * meanActive;
    if(start_len[0] >= thre) return;

    idx += start_len[0];
    if(start_len[0] + cnt >= thre) cnt = thre - start_len[0];

    for(int k = 0; k < cnt; k++){
        idx[k] = idx_temp[k];
    }
}


int ballquery_batch_p_cuda(int n, int meanActive, float radius, const float *xyz, const int *batch_idxs, const int *batch_offsets, int *idx, int *start_len, cudaStream_t stream) {
    // param xyz: (n, 3)
    // param batch_idxs: (n)
    // param batch_offsets: (B + 1)
    // output idx: (n * meanActive) dim 0 for number of points in the ball, idx in n
    // output start_len: (n, 2), int

    cudaError_t err;

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    int cumsum = 0;
    int* p_cumsum;
    cudaMalloc((void**)&p_cumsum, sizeof(int));
    cudaMemcpy(p_cumsum, &cumsum, sizeof(int), cudaMemcpyHostToDevice);

    ballquery_batch_p_cuda_<<<blocks, threads, 0, stream>>>(n, meanActive, radius, xyz, batch_idxs, batch_offsets, idx, start_len, p_cumsum);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaMemcpy(&cumsum, p_cumsum, sizeof(int), cudaMemcpyDeviceToHost);
    return cumsum;
}