#ifndef MC_H
#define MC_H
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector_types.h>
#include "global.h"
#include <stdio.h>

extern struct options{
    int subNum;
    char subFile[SUBNUMMAX][BUFFER];
    int flowNum;
    int* flowEns;    //flow ensample
    float* flowT;   //flow temperature
    float* flowN;   //flow density [mol/l]
    float* flowP;   //flow pressure [bar]
    int* flowIns;
    float** flowX;
    int mixRule;    //mixrule
    int singleXDim; //y dimension of threads
    int singleYDim; //y dimension of threads
    
    //
    int potNum; //numbers of potential parameters
} config;

extern struct molecules{
    int atomNum;
    char** aName;
    unsigned int* aType;
    float* x;
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
} initial;

/*extern struct flows{*/
/*    int molNum;*/
/*    float* xm;  //molecule coords*/
/*    float* ym;*/
/*    float* zm;*/
/*    float** xa;  //atom coords*/
/*    float** ya;*/
/*    float** za;*/
/*    int vaporNum;   //molecules in vapor phase*/
/*    int liquidNum;  //molecules in liquid phase*/
/*    int* vaporList; //list of molecules in vapor phase*/
/*    int* liquidList; //list of molecules in liquid phase*/
/*} initialFlows;*/

/*extern struct SingleBox{*/
/*    int molNum;*/
/*    float* xm;*/
/*    float* ym;*/
/*    float* zm;*/
/*    float** xa;*/
/*    float** ya;*/
/*    float** za;*/
/*} gpuSingleBox;*/

typedef struct{
    int molNum; //total number of molecules
    int* typeMolNum;    //numbers of molecule of each types
    int* type;  //type of i molecule
    float* xm;  //coordinats of molecules
    float* ym;
    float* zm;
    int* aNum;  //atom numbers
    int** aType;    //atomtype
    float** xa; //coordinats of atoms
    float** ya;
    float** za;
    float boxLen;
} singleBox;

typedef struct{
    char aName[5];
    float sigma;
    float epsilon;
    float alpha;
    float mass;
    float charge;
} potentialParam;

typedef struct{
    float sigma;
    float epsilon;
    float alpha;
    float charge;
} mixParam;

//GPU structures
typedef struct{
    float* xm;
    float* ym;
    float* zm;
    unsigned int* molNum;    //numbers of molecules
    unsigned int* typeMolNum;    //numbers of each type of molecule
    unsigned int* mType;
    //
    float* xa;
    float* ya;
    float* za;
    unsigned int* aType;

} gSingleBox;

typedef struct{ //topology
    unsigned int* aNum;  //numbers of atom in molecule
    unsigned int* aType;    //type of atom in molecule
    float* sigma;
    float* epsi;
    float* charge;
    float* mass;
//    float alpha;
    //bonds
    
    //angles
    
    //torsion
    
    
} gMolecula;

typedef struct{ //config
    unsigned int subNum;
    unsigned int flowNum;
    float* Temp;
} gOptions;

//    int vaporNum;   //molecules in vapor phase
//    int liquidNum;  //molecules in liquid phase
//    int* vaporList; //list of molecules in vapor phase
//    int* liquidList; //list of molecules in liquid phase
//} initialFlows;

extern int deviceCount;
extern cudaDeviceProp* deviceProp;  //array of device properties

extern FILE* logFile;

//functions
//initial
int get_device_prop(int &deviceCount, cudaDeviceProp* &deviceProp);
int read_options(options &config);
int read_init_gro(options config, molecules* &initial);
//write
int write_prop_log(int deviceCount, cudaDeviceProp* deviceProp,FILE* logFile);
int write_config_log(options con,FILE* logFile);

//flows
int initial_flows(options &config, singleBox* &initFlows,molecules* initMol, singleBox* &gpuSingleBox, int lines, potentialParam* allParams, potentialParam* gpuParams,potentialParam* &hostParams, mixParam** gpuMixParams, mixParam** hostMixParams,cudaDeviceProp* deviceProp);
int freeAll(singleBox* &gpuSingleBox,singleBox* &initFlows,options config);
int read_top(potentialParam* &allParams,int &lines);
char* remove_space(char* input);
int text_left(char* in, char* &out);
int data_to_device(gSingleBox &gBox, singleBox* &inputData, gOptions gConf, options &config, gMolecula gTop, potentialParam* Param, molecules* initMol);



__global__ void single_calc(singleBox* gpuSingleBox, potentialParam* gpuParams,int yDim, mixParam** gpuMixParams);
__device__ int single_calc_totenergy(int yDim, singleBox* gpuSingleBox, potentialParam* gpuParams, mixParam** gpuMixParams);
__device__ void single_calc_one_potential(int a, int b, singleBox* gpuSingleBox, potentialParam* gpuParams, mixParam** gpuMixParams);
__device__ void intra_potential(int a, int b, singleBox* gpuSingleBox, potentialParam* gpuParams, mixParam** gpuMixParams);
__device__ void inter_potential();
#endif
