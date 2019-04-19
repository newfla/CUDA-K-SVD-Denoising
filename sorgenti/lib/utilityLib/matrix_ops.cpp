#include <utilityLib.h>

using namespace utl;

MatrixOps::MatrixOps(){}

//**************************************
//  Destructor
//  Free Matrix* acquired (HOST/DEVICE) 
//*************************************
MatrixOps::~MatrixOps(){
    delete c;
}

//**************************************
//  Set coeff alfa e beta
//  input: alfa, beta (float)
//*************************************

void MatrixOps::setCoeff(float alfa, float beta){
    
    this->alfa = alfa;
    this->beta = beta;
}

//******************************************************************************************************
//  MatrixMult Factory method instantiates an object based on type
//  input:  + type (MatrixOpsType) of MatrixOps that will be used for the specified opstType operation
//  output: + engine (MatrixOps*)
//*****************************************************************************************************
MatrixOps* MatrixOps::factory(MatrixOpsType type){
    
    MatrixOps* mult;

    switch (type)
    {
        case CUBLAS_MULT:
            mult = new CuBlasMatrixMult();
            break;

        case CUBLAS_ADD:
            mult = new CuBlasMatrixAdd();
            break;
    
        default:
            return NULL;
    }
    return mult;
}

//*********************************************
//  Obtain time stat
//  output:  + timers (TimeElapsed*) ms timers
//********************************************
TimeElapsed* MatrixOps::getTimeElapsed(){
    return timeElapsed;
}