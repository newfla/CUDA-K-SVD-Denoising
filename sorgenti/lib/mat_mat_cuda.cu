// Numero di thread per riga di un blocco
#define BLOCKROW 32 

// Numero di thread per colonna di un blocco
#define BLOCKCOL 32 


__global__ void mat_mat_cuda_kernel(int n, int m, int p, int lda, int ldb, int ldc,  double *A, double *B, double *C){

    double prod=0; 

    int offset_i, offset_j; 
	
	// Offset di riga e colonna del thread rispetto a C
    offset_i = blockIdx.y*blockDim.y + threadIdx.y;
    offset_j = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Se non sono un thread in surplus 
    if(offset_i<n && offset_j<p){ 
        A=A+(lda*offset_i); // Equivalente a A=(double (*)[])&(A[offset_i][0]);
        B=B+offset_j;       // Equivalente a B=(double (*)[])&(B[0][offset_j]);
        for (int i = 0; i < m; ++i)
           prod += A[i] * B[ldb*i]; // Equivalente a prod+=A[0][i]*B[i][0];
	C[(offset_i*ldc)+offset_j]=prod;
	}
}

void mat_mat_cuda(int n, int m, int p, int lda, int ldb, int ldc, double *A, double *B, double *C){
	
    // Variabili del device
    double *A_d,*B_d,*C_d;

	// Allocazione matrici sul device
    cudaMalloc((void**)&A_d, n*m*sizeof(double));
    cudaMalloc((void**)&B_d, m*p*sizeof(double));
    cudaMalloc((void**)&C_d, n*p*sizeof(double));

	// Copia delle matrici dall'Host al Device
    cudaMemcpy(A_d,A,n*m*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,A,m*p*sizeof(double),cudaMemcpyHostToDevice);

    // Avrò bisogno di un thread per ogni elemento della matrice C
    int totThreads = n*p;
	
	// In base al numero max di thread per blocco calcolo il numero di blocchi necessari
    int totBlocks = (int) ceil(totThreads / BLOCKCOL*BLOCKROW);

	// Numero di righe/colonne nella griglia dei blocchi
    int gridRow = (int) ceil(n / (double) BLOCKROW); 
    int gridCol = (int) ceil(p / (double) BLOCKCOL);
	
	// Istanzio le dimensioni per la griglia e per ogni singolo blocco
    dim3 block(BLOCKCOL, BLOCKROW, 1);
    dim3 grid(gridCol, gridRow);

    //Calcolo su GPU
    mat_mat_cuda_kernel << < grid, block >> > (n, m, p, lda, ldb, ldc, A_d, B_d, C_d);

    //Copio il risultato sull'host
    cudaMemcpy(C, C_d, n*p*sizeof(double), cudaMemcpyDeviceToHost);

    //Libero memoria sul device
    cudaFree(A_d);
    cudaFree(B_d);
}