#include <iostream>
#include <svdLib.h>
using namespace svd;

std::vector<SvdContainer*> buildExample(int many, int m, int n){

    std::vector<SvdContainer*> list;

    for(int i = 0; i < many; i++)
    {
        SvdContainer* container = new SvdContainer(SvdEngine::factory(CUSOLVER_DN_DGESVD));
        container->setMatrix(Matrix::randomMatrix(m,n,m));
        list.push_back(container);
    }

    return list;
}

void cleanUp(std::vector<SvdContainer*> list){
    for(SvdContainer* pointer: list)
        delete pointer;
}

int main(int argc, char *argv[]) {

    std::vector<SvdContainer*> list = buildExample(2, 10, 10);

    std::cout<<"-----------------------------------------------"<<std::endl;

    for(int i = 0; i<list.size(); i++){
        SvdContainer* pointer = list[i];
        pointer->getOutputMatrices();
        TimeElapsed* timeElapsed = pointer->getTimeElapsed();

        std::cout<<"Matrix #"<<i+1<<std::endl;
        std::cout<<"Init Time: " << timeElapsed->getInitTime()<<"ms"<<std::endl;
        std::cout<<"Working Time: " << timeElapsed->getWorkingTime()<<"ms"<<std::endl;
        std::cout<<"Finalize Time: " << timeElapsed->getFinalizeTime()<<"ms"<<std::endl;
        std::cout<<"-----------------------------------------------"<<std::endl;
    }
    
    cleanUp(list);
    return 0;
}
