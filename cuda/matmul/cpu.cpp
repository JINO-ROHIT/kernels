int main(){
    const int M = 10, K = 2, N = 5;
    auto A = new int[M][K];
    auto B = new int[K][N];
    auto C = new int[M][N];

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float tmp = 0.0f;
            for (int k = 0; k < K; k++) {
                tmp += A[m][k] * B[k][n];
            }
            C[m][n] = tmp;
        }
    }

}