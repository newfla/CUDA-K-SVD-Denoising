#include <matUtilityLib.h>

using namespace baseUtl;
using namespace svd;
using namespace matUtl;
using namespace thrust;
using namespace thrust::placeholders;

CuBlasMatrixOmp::CuBlasMatrixOmp(){}

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

    sparseCode = new device_vector<float>(b->n * a->n);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->init = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}

//******************************
// Clear cuBlas additional data
//*****************************
void CuBlasMatrixOmp::finalize(){

    auto start = std::chrono::steady_clock::now();

    c = new Matrix(a->n, b->n, a->n, sparseCode);
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
    
    float norm = 0;
    int max = 0, min = 0, chosenAtomIdx = 0;

    for(int inputIdx = 0; inputIdx < patchesMatrix->n; inputIdx++){ //n == #columns == # patches

        device_vector<float> residualVec(patchesMatrix->deviceVector->begin() + (inputIdx * patchesMatrix->m),
                                     patchesMatrix->deviceVector->begin() + ((inputIdx+1) * patchesMatrix->m));
        
        device_vector<float> tempVec(dictionaryMatrix->m);//,0);
        device_vector<float> thisSparseCode(dictionaryMatrix->n);//,0);
        device_vector<int> chosenAtomIdxList;
		device_vector<float> chosenAtomList; 
        int iter = 0;
        
        while(iter < maxIters){
        
            device_vector<float> proj(dictionaryMatrix->n);//,0);

            cublasSgemv(handle,
                        CUBLAS_OP_T,
                        dictionaryMatrix->m,
                        dictionaryMatrix->n,
                        &alfa,
                        raw_pointer_cast(dictionaryMatrix->deviceVector->data()),
                        dictionaryMatrix->ld,
                        raw_pointer_cast(residualVec.data()),
                        1,
                        &beta,
                        raw_pointer_cast(proj.data()),
                        1);

            cublasIsamax(handle,
                         dictionaryMatrix->n,
                         raw_pointer_cast(proj.data()),
                         1,
                         &max);
            max--;

            cublasIsamin(handle,
                         dictionaryMatrix->n,
                         raw_pointer_cast(proj.data()),
                         1,
                         &min);
            min--;

           /* chosenAtomIdx = max;
    
            if(abs(proj[max]) < abs(proj[min]))
                chosenAtomIdx = min;*/
            
            chosenAtomIdx = (abs(proj[max]) > abs(proj[min])) ? max : min;

          
            chosenAtomIdxList.push_back(chosenAtomIdx);

            chosenAtomList.insert(chosenAtomList.end(),
                                  dictionaryMatrix->deviceVector->begin() + (chosenAtomIdx * dictionaryMatrix->m),
                                  dictionaryMatrix->deviceVector->begin() + ((chosenAtomIdx + 1) * dictionaryMatrix->m));
            
            //ChosenAtomList Pseudo-Inverse
            device_vector<float>* copiedList = new device_vector<float> (chosenAtomList.begin(), chosenAtomList.end());
            Matrix* toPinvert = new Matrix(dictionaryMatrix->m, chosenAtomIdxList.size(), dictionaryMatrix->m, copiedList);

            SvdContainer* pointer1 = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVDJ));
            pointer1->setMatrix(toPinvert);
            host_vector<Matrix*> usv = pointer1->getDeviceOutputMatrices();

            device_vector<float> sVector(dictionaryMatrix->m * chosenAtomIdxList.size(),0);            
            device_vector<float> tempMatMult(chosenAtomIdxList.size() * dictionaryMatrix->m);
            device_vector<float> pseudoInverse(chosenAtomIdxList.size() * dictionaryMatrix->m);

            for(int i = 0; i < usv[1]->n; i++){    
                if(usv[1]->deviceVector->data()[i] != 0)
                    sVector[(i * chosenAtomIdxList.size()) + i] = 1./usv[1]->deviceVector->data()[i];
            }

            cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N, //sVector è già trasposto a mano in realtà
                usv[2]->m,
                usv[0]->m,
                usv[2]->n,
                &alfa,
                raw_pointer_cast(usv[2]->deviceVector->data()),
                usv[2]->ld,
                raw_pointer_cast(sVector.data()),
                usv[2]->ld,
                &beta,
                raw_pointer_cast(tempMatMult.data()),
                usv[2]->ld);

            cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                usv[2]->m,
                usv[0]->m,
                usv[0]->m,
                &alfa,
                raw_pointer_cast(tempMatMult.data()),
                usv[2]->ld,
                raw_pointer_cast(usv[0]->deviceVector->data()),
                usv[0]->ld,
                &beta,
                raw_pointer_cast(pseudoInverse.data()),
                usv[2]->ld);

            device_vector<float> weightList(chosenAtomIdxList.size());

            if(chosenAtomIdxList.size() == 1)
                weightList[0] = inner_product(pseudoInverse.begin(),
                                pseudoInverse.end(),
                                patchesMatrix->deviceVector->begin() + (inputIdx * patchesMatrix->m),
                                0.f);
            else{
                cublasSgemv(handle,
                            CUBLAS_OP_N,
                            chosenAtomIdxList.size(),
                            dictionaryMatrix->m,
                            &alfa,
                            raw_pointer_cast(pseudoInverse.data()),
                            chosenAtomIdxList.size(),
                            raw_pointer_cast(patchesMatrix->deviceVector->data()) + (inputIdx * patchesMatrix->m),
                            1,
                            &beta,
                            raw_pointer_cast(weightList.data()),
                            1);            
            }
            //store coefficient 
            for(int i = 0 ; i < chosenAtomIdxList.size() ; i++){
				int thisAtomIdx = chosenAtomIdxList[i] ;
				thisSparseCode[thisAtomIdx] = weightList[i];
                
			}
          
            if(weightList.size()==1)
                transform(chosenAtomList.begin(),
                          chosenAtomList.end(),
                          tempVec.begin(),
                          _1*weightList[0]);
            else{
              /*  std::cout<<"chosenAtomListSize"<<chosenAtomList.size()<<std::endl;
                std::cout<<"weightListSize"<<weightList.size()<<std::endl;
                std::cout<<"tempVecSize"<<tempVec.size()<<std::endl;
                std::cout<<std::cin.get();
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
                            1);*/
                transform(chosenAtomList.begin(),
                          chosenAtomList.begin() + dictionaryMatrix->m,
                          tempVec.begin(),
                          _1 * weightList[0]); 
                
                device_vector<float> tempVec2(dictionaryMatrix->m);//,0);

                for(int i = 1; i < weightList.size(); i++)
                {
                    transform(chosenAtomList.begin() + (i * dictionaryMatrix->m),
                              chosenAtomList.begin() + ((i + 1) * dictionaryMatrix->m),
                              tempVec2.begin(),
                              _1 * weightList[i]);
                    
                
                    
                    transform(tempVec2.begin(),
                              tempVec2.end(),
                              tempVec.begin(),
                              tempVec.begin(),
                              plus<float>());
                }
                            
            }

            transform(patchesMatrix->deviceVector->begin() + (inputIdx * patchesMatrix->m),
                      patchesMatrix->deviceVector->begin() + ((inputIdx+1) * patchesMatrix->m),
                      tempVec.begin(),
                      residualVec.begin(),
                      minus<float>());
            
            cublasSnrm2(handle,
                    dictionaryMatrix->m,
                    raw_pointer_cast(residualVec.data()),
                    1,
                    &norm);

            for(int i=0; i<usv.size(); i++)
                delete usv[i]; 
            
            if(norm < 0.001) break;
            
            iter++;
        }
        sparseCode->insert(sparseCode->begin() + (inputIdx * dictionaryMatrix->n), thisSparseCode.begin(), thisSparseCode.end());
    }
    /*int x=0, y=0;
    for(int i = 0; i < sparseCode->size(); i++)
    {
        if(sparseCode->data()[i]==0)
            x++;
        else 
            y++;
    }
    std::cout<<"nulli: "<<x<<" non nulli: "<<y<<std::endl;
    std::cin.get();*/
    
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
        iterEpsilon = sqrtf( 1e-4 * normi);
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