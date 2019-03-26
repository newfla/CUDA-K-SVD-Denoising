#include <svdLib.h>

using namespace svd;

Matrix::~Matrix(){
    std::cout<<"\nMatrix\n";
    delete[] matrix;
}

Matrix* Matrix::randomMatrix(int m, int n, int ld){
            Matrix * out = new Matrix();

            out->m = m;
            out->n = n;
            out ->ld = ld;
            out->matrix = new double[m*n]();

            std::random_device rd;
            std::mt19937 engine(rd());
            std::uniform_real_distribution<float> dist(0.,1.);

            for(int i = 0; i < m*n; i++)
              out->matrix[i] = dist(engine);

            return out;
        }
