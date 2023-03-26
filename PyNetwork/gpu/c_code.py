c_code = """
    #define BLOCK_SIZE 2
    #define A_BLOCK_STRIDE (BLOCK_SIZE * a_width)
    #define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_height)

    __kernel void ew_add(__global float *A, __global float *B, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        out[index] = A[index] + B[index];
    }

    __kernel void ew_sub(__global float *A, __global float *B, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        out[index] = A[index] - B[index];
    }

    __kernel void ew_div(__global float *A, __global float *B, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        out[index] = A[index] / B[index];
    }

    __kernel void ew_mul(__global float *A, __global float *B, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        out[index] = A[index] * B[index];
    }
    
    __kernel void ew_greater(__global float *A, __global float *B, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        out[index] = A[index] > B[index];
    }    

    __kernel void ew_equal(__global float *A, __global float *B, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        out[index] = (A[index] == B[index]);
    }

    __kernel void ew_sign(__global float *A, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        if (A[index] > 0) {
            out[index] = 1.0;
        } else if (A[index] == 0) {
            out[index] = 0.0;
        }
        else{
            out[index] = -1.0;
        }
    }

    __kernel void reduce(__global float *a,
            __global float *r,
            int length1,
            int length2){
        int gj = get_global_id(0); //along row | col number
        float sum = 0.0f;
        for (int k = 0; k < length2; k++){
            sum += a[gj + length1 * k];
        }
        r[gj] = sum;  
    }

    __kernel void matrixmultiply2dlocal(int heightA, int widthA, int heightB, int widthB, __global float *A, __global float *B, __global float *out){
        int local_row = get_local_id(1);
        int local_col = get_local_id(0);

        int row = get_global_id(1);
        int col = get_global_id(0);

        __local float Ab[BLOCK_SIZE][BLOCK_SIZE];
        __local float Bb[BLOCK_SIZE][BLOCK_SIZE];

        int num = widthA / BLOCK_SIZE;
        float sum = 0.0f;
        
        for (int n=0; n < num; n++){
            int row_b = n*BLOCK_SIZE + local_row;
            int col_b = n*BLOCK_SIZE + local_col;

            Ab[local_row][local_col] = A[row*widthA + col_b];
            Bb[local_row][local_col] = B[row_b*widthB + col];

            barrier(CLK_LOCAL_MEM_FENCE);  

            for (int i=0; i < BLOCK_SIZE; i++){
                sum += Ab[local_row][i] * Bb[i][local_col];
            }  

            barrier(CLK_LOCAL_MEM_FENCE);    
        }

        out[row*widthB + col] = sum;
    }

    __kernel void naive_transpose(__global float *a_t, __global float *a, int width, int height){
        int read_idx = get_global_id(0) + get_global_id(1) * width;
        int write_idx = get_global_id(1) + get_global_id(0) * height;
        a_t[write_idx] = a[read_idx];

    }

    __kernel void transpose(__global float *a_t, __global float *a, int a_width, int a_height, __local float *a_local){
        int base_idx_a   =
            get_group_id(0) * BLOCK_SIZE +
            get_group_id(1) * A_BLOCK_STRIDE;
        int base_idx_a_t =
            get_group_id(1) * BLOCK_SIZE +
            get_group_id(0) * A_T_BLOCK_STRIDE;
        int glob_idx_a   =
            base_idx_a + get_local_id(0) + a_width * get_local_id(1);
        int glob_idx_a_t =
            base_idx_a_t + get_local_id(0) + a_height * get_local_id(1);
        a_local[get_local_id(1)*BLOCK_SIZE+get_local_id(0)] = a[glob_idx_a];
        barrier(CLK_LOCAL_MEM_FENCE);
        a_t[glob_idx_a_t] = a_local[get_local_id(0)*BLOCK_SIZE+get_local_id(1)];
}

    __kernel void repeat(__global float *out, __global float *A, int width){
        int index = get_global_id(0);
        int Arow = index / width;

        out[index] = A[Arow];
    }
"""