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

void testBatchDenoiser(){
    BatchDenoiser* batchDenoiser = BatchDenoiser::factory(CUDA_K_GESVDJ, "/home/fbizzarri/prova/config.json");
    batchDenoiser->seqBatchDenoising();

    host_vector<baseUtl::TimeElapsed*> times = batchDenoiser->getTimeElapsed();

    host_vector<host_vector<double>*> psnr = batchDenoiser->getPsnr(); 

    double init = times[0]->init, work = times[0]->working, fin = times[0]->finalize, tot = times[0]->getTotalTime();

    init/=1000.;
    work/=1000.;
    fin/=1000.;
    tot/=1000.;

    std::cout<<std::endl<<"# Total Batch execution time: "<<tot<<" s"<<std::endl;
    std::cout<<"    Total Batch init time: "<<init<<" s"<<std::endl;
    std::cout<<"    Total Batch working time: "<<work<<" s"<<std::endl;
    std::cout<<"    Total Batch finalize time: "<<fin<<" s"<<std::endl;

    for(int i = 1; i < times.size(); i++)
    {
        init = times[i]->init;
        work = times[i]->working;
        fin = times[i]->finalize;
        tot = times[i]->getTotalTime();

        work/=1000.;
        tot/=1000.;

        std::cout<<std::endl<<"## Image: "<<i<<" execution time: "<<tot<<" s"<<std::endl;
        std::cout<<"    init time: "<<init<<" ms"<<std::endl;
        std::cout<<"    working time: "<<work<<" s"<<std::endl;
        std::cout<<"    finalize time: "<<fin<<" ms"<<std::endl;
        if(psnr[i-1]->data()[0]>=0){
            std::cout<<"    PSNR before:"<<psnr[i-1]->data()[0]<<std::endl;
            std::cout<<"    PSNR after:"<<psnr[i-1]->data()[1]<<std::endl;
        }        
    }
    


    delete batchDenoiser;
}

int main(int argc, char *argv[]) {

    /*    

    testCuSolverSVD(1024,512,10);
    testCuSolverSVD(2048,1024,10);
    testCuSolverSVD(4096,2048,10);
    testCuSolverSVD(8192,4096,5);

    */

   testBatchDenoiser();
   
    return 0;
}
