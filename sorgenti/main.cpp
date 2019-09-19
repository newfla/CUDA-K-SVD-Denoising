#include <denoisingLib.h>

using namespace denoising;
using namespace svd;
using namespace baseUtl;
using namespace thrust;
using namespace cimg_library;

void testCuSolverSVD(int m, int n, int tot){
    std::cout<<"---------------------<<"<<m<<","<<n<<">>--------------------------"<<std::endl;

    for(int i = 0; i<tot; i++){

        std::cout<<std::endl<<"Matrix #"<<i+1<<std::endl;

        Matrix* input = Matrix::randomMatrixHost(m,n,m);
        Matrix * input1 = input->cloneHost();
        
        SvdContainer* pointer = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVD));
        pointer->setMatrix(input);
        Matrix* matrixS = pointer->getOutputMatrices()[1];
        TimeElapsed* timeElapsed = pointer->getTimeElapsed();

        std::cout<<"------------------QR--------------------"<<std::endl;
        std::cout<<"Init Time: " << timeElapsed->init<<"ms"<<std::endl;
        std::cout<<"Working Time: " << (((double)timeElapsed->working)/1000.)<<"s"<<std::endl;
        std::cout<<"Finalize Time: " << timeElapsed->finalize<<"ms"<<std::endl;
        std::cout<<std::endl;

        SvdContainer* pointer1 = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVDJ));
        pointer1->setMatrix(input1); 
        std::cout<<"------------------JACOBI--------------------"<<std::endl;

        Matrix* matrix1S = pointer1->getOutputMatrices()[1];
        timeElapsed = pointer1->getTimeElapsed();

        std::cout<<"Init Time: " << timeElapsed->init<<"ms"<<std::endl;
        std::cout<<"Working Time: " << (((double)timeElapsed->working)/1000.)<<"s"<<std::endl;
        std::cout<<"Finalize Time: " << timeElapsed->finalize<<"ms"<<std::endl;

        float maxError = 0;

        for(int j = 0; j < n; j++){
            float err = fabs( matrix1S->hostVector->data()[j] - matrixS->hostVector->data()[j] );
            maxError = (maxError > err) ? maxError : err;
        }

        std::cout<<"Max error: "<<maxError<<std::endl;
        std::cout<<"-----------------------------------------------"<<std::endl<<std::endl;

        delete pointer;
        delete pointer1;
    }
}

void testBatchDenoiser(std::string file){
    
    double init = 0, work = 0, fin = 0;
    BatchDenoiser* batchDenoiser = BatchDenoiser::factory(file);
    host_vector<Denoiser*> list = batchDenoiser->getDenoiserList();

    for (int i = 0; i < list.size(); i++){

        Denoiser* denoiser = list[i];
        TimeElapsed* times;
        host_vector<double> *psnr;

        char status = denoiser->denoising();
        switch (status){
            case 0:
            
                times = denoiser->getTimeElapsed();
                psnr = denoiser->getPsnr();
                init += times->init;
                work += times->working;
                fin += times->finalize;

                std::cout<<std::endl<<"## Image: "<<i+1<<" execution time: "<<times->getTotalTime()/1000.<<" s"<<std::endl;
                std::cout<<"    init time: "<<times->init<<" ms"<<std::endl;
                std::cout<<"    working time: "<<times->working/1000.<<" s"<<std::endl;
                std::cout<<"    finalize time: "<<times->finalize<<" ms"<<std::endl;

                if(psnr->data()[0]>=0){
                std::cout<<"    PSNR before:"<<psnr->data()[0]<<std::endl;
                std::cout<<"    PSNR after:"<<psnr->data()[1]<<std::endl;
                }
                break;
            
            case -1:
                std::cout<<"Image Loading Failed"<<std::endl;
                break;
            
            case -2:
                std::cout<<"Denoising Failed"<<std::endl;
                break;

            case -3:
                std::cout<<"Image Saving Failed"<<std::endl;
                break;
            }
    }

    init/=1000.;
    work/=1000.;
    fin/=1000.;

    std::cout<<std::endl<<"# Total Batch execution time: "<<init + fin + work<<" s"<<std::endl;
    std::cout<<"    Total Batch init time: "<<init <<" s"<<std::endl;
    std::cout<<"    Total Batch working time: "<<work<<" s"<<std::endl;
    std::cout<<"    Total Batch finalize time: "<<fin<<" s"<<std::endl;
    delete batchDenoiser;
}

int main(int argc, char *argv[]) {

    std::string file = "../config.json";

    if(argc == 2)
        file = argv[1];

    testBatchDenoiser(file);
   
    return 0;
}
