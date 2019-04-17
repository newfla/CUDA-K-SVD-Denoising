#include <denoisingLib.h>

using namespace utl;

//**************************************
//  Destructor
//  Free Matrix* acquired (HOST/DEVICE) 
//*************************************
MatrixMult::~MatrixMult(){
    delete a;
    delete b;
    delete c;
}

//****************************************************************************
//  MatrixMult Factory method instantiates an object based on type
//  input:  + type (MatrixMultType) of MatrixMult that will be used for alfa*op( A )*op( B ) + beta*C,
//  output: + engine (MatrixMult*)
//***************************************************************************

MatrixMult* MatrixMult::factory(MatrixMultType type, utl::Matrix* a, utl::Matrix* b,int alfa, int beta){
    
    MatrixMult* mult;

    switch (type)
    {
        case CUBLASS_MULT:
            mult = new CuBlassMatrixMult();
            break;
    
        default:
            return NULL;
    }

    mult->a = a;
    mult->b = b;
    mult->alfa = alfa;
    mult->beta = beta;

    return mult;
}

utl::TimeElapsed* MatrixMult::getTimeElapsed(){
    return timeElapsed;
}