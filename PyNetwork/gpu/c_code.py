c_code = """
    #define BLOCK_SIZE 16

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

    __kernel void transpose(__global float *a_t, __global float *a, int width, int height, __local float *a_local){
        int global_col = get_global_id(0);
        int global_row = get_global_id(1);

        int local_col = get_local_id(0);
        int local_row = get_local_id(1);

        int local_index = local_row * BLOCK_SIZE + local_col;

        a_local[local_index] = a[global_row * width + global_col];

        barrier(CLK_LOCAL_MEM_FENCE);

        int group_col = get_group_id(0);
        int group_row = get_group_id(1);

        /* Transpose the blocks */
        global_row = group_col * BLOCK_SIZE + local_row;
        global_col = group_row * BLOCK_SIZE + local_col;

        a_t[global_row * height + global_col] = a_local[local_col * BLOCK_SIZE + local_row];
    }
"""