#include <iostream>
#include <svdLib.h>
using namespace svd;

SvdContainer* buildExample(int m, int n){

    SvdContainer* container = new SvdContainer(SvdEngine::factory(CUSOLVER_DN_DGESVD));
    container->setMatrix(Matrix::randomMatrix(m,n,m));

    return container;
}

int main(int argc, char *argv[]) {

    std::cout<<"-----------------------------------------------"<<std::endl;

    for(int i = 0; i<10; i++){
        SvdContainer* pointer = buildExample(800, 600);
        pointer->getOutputMatrices();
        TimeElapsed* timeElapsed = pointer->getTimeElapsed();

        std::cout<<"Matrix #"<<i+1<<std::endl;
        std::cout<<"Init Time: " << timeElapsed->getInitTime()<<"ms"<<std::endl;
        std::cout<<"Working Time: " << timeElapsed->getWorkingTime()<<"ms"<<std::endl;
        std::cout<<"Finalize Time: " << timeElapsed->getFinalizeTime()<<"ms"<<std::endl;
        std::cout<<"-----------------------------------------------"<<std::endl;

        delete pointer;
    }
    return 0;
}
