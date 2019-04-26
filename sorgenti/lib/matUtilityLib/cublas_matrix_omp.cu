#include <matUtilityLib.h>

using namespace baseUtl;
using namespace svd;
using namespace matUtl;
using namespace thrust;
using namespace thrust::placeholders;

CuBlasMatrixOmp::CuBlasMatrixOmp(){}

//**************************************
//  Set eps_float maxIter
//  input: epsilon maxIter (float, int as float)
//*************************************
void CuBlasMatrixOmp::setLimits(float epsilon, int maxIters){
    
    this->epsilon = epsilon;
    this->maxIters = (int) maxIters;
}

//********************************************************
// Create cuBlas additional data and move data on DEVICE
//*******************************************************
void CuBlasMatrixOmp::init(){

    auto start = std::chrono::steady_clock::now();

    cublasCreate(&handle);

    if(a->deviceVector == NULL)
        a->copyOnDevice();

    if(b->deviceVector == NULL)
        b->copyOnDevice();

    sparseCode = new device_vector<float>();

    auto end = std::chrono::steady_clock::now();
    timeElapsed->init = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}

//******************************
// Clear cuBlas additional data
//*****************************
void CuBlasMatrixOmp::finalize(){

    auto start = std::chrono::steady_clock::now();

    c = new Matrix(b->n, a->n, b->n, sparseCode);
    cublasDestroy(handle);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->finalize = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}


// **********************************************************************
// MATCHING PURSUIT ALGORITM (OMP) FOR SPARSE CODING 
// Input:  + patchesMatrix [feature per patch * number of patches] 
//         + dictionary   [feature * atoms]         
// Output: + sparse coding of input vector [ atoms * number of patches]
//**********************************************************************
baseUtl::Matrix* CuBlasMatrixOmp::work(Matrix* patchesMatrix, Matrix* dictionaryMatrix){

    this->b = patchesMatrix;
    this->a = dictionaryMatrix;

    init();

    auto start = std::chrono::steady_clock::now();

    float* noisePatches = raw_pointer_cast(patchesMatrix->deviceVector->data());
    float* dictionary = raw_pointer_cast(dictionaryMatrix->deviceVector->data());
    float* sparseCodeV = raw_pointer_cast(sparseCode->data());
    float norm = 0;
    int chosenAtomIdx = 0;

    for(int inputIdx = 0; inputIdx < patchesMatrix->n; inputIdx++){ //n == #columns == # patches

        device_vector<float> thisInput(patchesMatrix->deviceVector->begin() + (inputIdx * patchesMatrix->m),
                                     patchesMatrix->deviceVector->begin() + ((inputIdx+1) * patchesMatrix->m));
        
        device_vector<float> residualVec(thisInput.begin(), thisInput.end());
        device_vector<float> thisSparseCode(dictionaryMatrix->n,0) ;
        device_vector<int> chosenAtomIdxList;
		device_vector<float> chosenAtomList; 
        int iter = 0;

        while(iter < maxIters){

            std::cout<<"Iter OMP: "<<iter<<std::endl;
        
            device_vector<float> proj(dictionaryMatrix->n,0);

            host_vector<float> tt;

            cublasSgemv(handle,
                        CUBLAS_OP_T,
                        dictionaryMatrix->m,
                        dictionaryMatrix->n,
                        &alfa,
                        dictionary,
                        dictionaryMatrix->ld,
                        raw_pointer_cast(residualVec.data()),
                        1,
                        &beta,
                        raw_pointer_cast(proj.data()),
                        1);

            std::cout<<"calcolo del residuo iniziale\n";

            cublasIsamax(handle,
                         dictionaryMatrix->n,
                         raw_pointer_cast(proj.data()),
                         1,
                         &chosenAtomIdx);

            chosenAtomIdx--;

            std::cout<<"Becco atomo + grande:"<<chosenAtomIdx<<std::endl;

            device_vector<float> chosenAtom(dictionaryMatrix->deviceVector->begin() + (chosenAtomIdx * dictionaryMatrix->m),
                                  dictionaryMatrix->deviceVector->begin() + ((chosenAtomIdx +1) * dictionaryMatrix->m));

            chosenAtomIdxList.push_back(chosenAtomIdx);

            chosenAtomList.insert(chosenAtomList.end(), chosenAtom.begin(), chosenAtom.end());

            std::cout<<"creo copia di choosenAtomList perchè verrà modificata da SVD\n";

            device_vector<float>* copiedList = new device_vector<float> (chosenAtomList.begin(), chosenAtomList.end());
            Matrix * toPinvert = new Matrix( dictionaryMatrix->m, chosenAtomList.size() / dictionaryMatrix->m, dictionaryMatrix->m, copiedList);

            std::cout<<"lunghezza chhsoenAtomList: "<<chosenAtomList.size()<<std::endl;

            SvdContainer* pointer1 = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVDJ));
            pointer1->setMatrix(toPinvert);
            device_vector<Matrix*> usv = pointer1->getDeviceOutputMatrices();
            Matrix* u = usv[0];
            Matrix* s = usv[1];
            Matrix* v = usv[2]; 

            std::cout<<"Svd fatto\n";

            device_vector<float> sVector(u->m * v->n,0);            
            device_vector<float> tempMatMult(v->m * u->m);
            device_vector<float> pseudoInverse(v->m * u->m);

            for(int i = 0; i < s->m; i++){

                //if( s->deviceVector->data()[i] > 1e-4 && s->deviceVector->data()[i] != 0)
                    sVector[(i * u->m) + i] = 1.;//s->deviceVector->data()[i];

            }

            std::cout<<"Aggiornamento diagonale";

            cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                v->m,
                u->n,
                v->n,
                &alfa,
                raw_pointer_cast(v->deviceVector->data()),
                v->ld,
                raw_pointer_cast(sVector.data()),
                u->ld,
                &beta,
                raw_pointer_cast(tempMatMult.data()),
                b->ld);

            cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                v->m,
                u->m,
                u->n,
                &alfa,
                raw_pointer_cast(tempMatMult.data()),
                v->ld,
                raw_pointer_cast(sVector.data()),
                u->ld,
                &beta,
                raw_pointer_cast(pseudoInverse.data()),
                b->ld);

            device_vector<float> weightList(chosenAtomIdxList.size());

            cublasSgemv(handle,
                        CUBLAS_OP_N,
                        chosenAtomIdxList.size(),
                        dictionaryMatrix->m,
                        &alfa,
                        raw_pointer_cast(pseudoInverse.data()),
                        chosenAtomIdxList.size(),
                        raw_pointer_cast(thisInput.data()),
                        1,
                        &beta,
                        raw_pointer_cast(weightList.data()),
                        1);

            for(int i = 0 ; i < chosenAtomIdxList.size() ; i++){
				int thisAtomIdx = chosenAtomIdxList[i] ; 
				thisSparseCode[thisAtomIdx] = weightList[i];
			}

            device_vector<float> tempVec(dictionaryMatrix->m);

            cublasSgemv(handle,
                        CUBLAS_OP_N,
                        dictionaryMatrix->m,
                        chosenAtomList.size(),
                        &alfa,
                        raw_pointer_cast(chosenAtomList.data()),
                        dictionaryMatrix->m,
                        raw_pointer_cast(weightList.data()),
                        1,
                        &beta,
                        raw_pointer_cast(tempVec.data()),
                        1);

            transform(thisInput.begin(), thisInput.end(), tempVec.begin(), residualVec.begin(), minus<float>());
            
            cublasSnrm2(handle,
                    dictionaryMatrix->n,
                    raw_pointer_cast(residualVec.data()),
                    1,
                    &norm);

            if(norm < 0.001) break;
            
            iter++;
        }
    }

    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    finalize();

    return c;
}


baseUtl::Matrix* CuBlasMatrixOmp::work2(Matrix* dictionaryMatrix, Matrix* patchesMatrix){

    this->b = patchesMatrix;
    this->a = dictionaryMatrix;

    init();

    auto start = std::chrono::steady_clock::now();

    device_vector<float> residualThrust(a->n * b->n);
    float* noisePatches = raw_pointer_cast(a->deviceVector->data());
    float* dictionary = raw_pointer_cast(b->deviceVector->data());
    float* residual = raw_pointer_cast(residualThrust.data());
    float normi, normf, iterEpsilon, coff;
    int q;

    for(int i = 0; i < patchesMatrix->n; i++){ //n == #columns == # patches   

        cublasSgemv(handle,
                    CUBLAS_OP_T,
                    dictionaryMatrix->m,
                    dictionaryMatrix->n,
                    &alfa,
                    dictionary,
                    dictionaryMatrix->ld,
                    noisePatches + (i * dictionaryMatrix->m),
                    1,
                    &beta,
                    residual + (i * dictionaryMatrix->n),
                    1);
    }

    for(int i = 0; i < patchesMatrix->n; i++){ //n == #columns == # patches

        cublasSnrm2(handle,
                    dictionaryMatrix->n,
                    residual + (i * dictionaryMatrix->n),
                    1,
                    &normi);

        int iter = 0;
        iterEpsilon = sqrtf( epsilon * normi);
        normf = normi;
        while (normf > iterEpsilon && iter < maxIters){
            
          cublasSgemv(handle,
                    CUBLAS_OP_N,
                    dictionaryMatrix->m,
                    dictionaryMatrix->n,
                    &alfa,
                    dictionary,
                    dictionaryMatrix->ld,
                    residual + (i * dictionaryMatrix->n),
                    1,
                    &beta,
                    noisePatches + (i * dictionaryMatrix->m),
                    1);

            cublasIsamax(handle,
                        dictionaryMatrix->m,
                        noisePatches + (i * dictionaryMatrix->m),
                        1,
                        &q);

            q-=1;

            cublasGetVector(1, 
                            sizeof(coff),
                            noisePatches + q + (i * dictionaryMatrix->m),
                            1,
                            &coff,
                            1);
            
            coff = - coff;

            cublasSaxpy(handle,
                        dictionaryMatrix->n,
                        &coff,
                        dictionary + q,
                        dictionaryMatrix->m,
                        residual + (i * dictionaryMatrix->n),
                        1);

            cublasSnrm2(handle,
                        dictionaryMatrix->n,
                        residual + (i * dictionaryMatrix->n),
                        1,
                        &normf);
            
            iter++;
        }
        
    }
    

    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    finalize();

    return c;
}