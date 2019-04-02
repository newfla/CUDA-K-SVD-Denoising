#include <iostream>
#include<string>
#include <svdLib.h>
using namespace svd;

SvdContainer* buildExample(int m, int n, SvdEngineType type){

    SvdContainer* container = new SvdContainer(SvdEngine::factory(type));
    container->setMatrix(Matrix::randomMatrix(m,n,m));

    return container;
}

void exec(int m, int n, int tot){
    std::cout<<"---------------------<<"<<m<<","<<n<<">>--------------------------"<<std::endl;

    for(int i = 0; i<tot; i++){
        
        SvdContainer* pointer = buildExample(m, n, CUSOLVER_GESVDJ);
        pointer->getOutputMatrices();
        TimeElapsed* timeElapsed = pointer->getTimeElapsed();

        std::cout<<"Matrix #"<<i+1<<std::endl;
        std::cout<<"Init Time: " << timeElapsed->getInitTime()<<"ms"<<std::endl;
        std::cout<<"Working Time: " << (double)(timeElapsed->getWorkingTime())/1000.<<"s"<<std::endl;
        std::cout<<"Finalize Time: " << timeElapsed->getFinalizeTime()<<"ms"<<std::endl;
        std::cout<<"-----------------------------------------------"<<std::endl;

        delete pointer;
    }
}

void exec2(int m, int n, int tot){
    std::cout<<"---------------------<<"<<m<<","<<n<<">>--------------------------"<<std::endl;

    for(int i = 0; i<tot; i++){

        std::cout<<std::endl<<"Matrix #"<<i+1<<std::endl;

        Matrix* input = Matrix::randomMatrix(m,n,m);
        
        SvdContainer* pointer = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVD));
        pointer->setMatrix(input);
        Matrix* matrixS = pointer->getOutputMatrices()[1];
        TimeElapsed* timeElapsed = pointer->getTimeElapsed();

        std::cout<<"------------------QR--------------------"<<std::endl;
        std::cout<<"Init Time: " << timeElapsed->getInitTime()<<"ms"<<std::endl;
        std::cout<<"Working Time: " << (double)(timeElapsed->getWorkingTime())/1000.<<"s"<<std::endl;
        std::cout<<"Finalize Time: " << timeElapsed->getFinalizeTime()<<"ms"<<std::endl;
        std::cout<<std::endl;

        SvdContainer* pointer1 = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVDJ));
        pointer1->setMatrix(input); 

        std::cout<<"------------------JACOBI--------------------"<<std::endl;

        Matrix* matrix1S = pointer1->getOutputMatrices()[1];
        timeElapsed = pointer1->getTimeElapsed();

        std::cout<<"Init Time: " << timeElapsed->getInitTime()<<"ms"<<std::endl;
        std::cout<<"Working Time: " << (double)(timeElapsed->getWorkingTime())/1000.<<"s"<<std::endl;
        std::cout<<"Finalize Time: " << timeElapsed->getFinalizeTime()<<"ms"<<std::endl;

        float maxError = 0;

        for(int j = 0; j < n; j++){
            float err = fabs( matrix1S->matrix[j] - matrixS->matrix[j] );
            maxError = (maxError > err) ? maxError : err;
        }

        std::cout<<"Max error: "<<maxError<<std::endl;
        std::cout<<"-----------------------------------------------"<<std::endl<<std::endl;

        //Comment line #7 in svd_engine.cpp
        delete pointer;
        delete pointer1;
        delete input; 
    }
}

int main(int argc, char *argv[]) {

    int m = 800, n = 800;

    if(argc==3){
        m = std::stoi(argv[1]);
        n = std::stoi(argv[2]);
    }

    exec2(1024,512,10);
    exec2(2048,1024,10);
    exec2(4096,2048,10);
    exec2(8192,4096,5);

    return 0;
}
