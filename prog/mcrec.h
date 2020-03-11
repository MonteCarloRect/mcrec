#ifndef MC_H
#define MC_H

#include <cuda_runtime.h>

#include <cuda.h>
#include <vector_types.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*#include <curand_kernel.h>*/
/*#include <curand.h>*/

//#include <curand_kernel.h>
//#include <curand.h>

#include "global.h"


extern struct options{
    int subNum;
    char subFile[SUBNUMMAX][BUFFER];
    int flowNum;
    int* flowEns;    //flow ensample
    float* flowT;   //flow temperature
    float* flowN;   //flow density [mol/l]
    float* flowP;   //flow pressure [bar]
    int** flowIns;  //flow molecules  inserted by substance
    float** flowX;
    int mixRule;    //mixrule
    int singleXDim; //y dimension of threads
    int singleYDim; //y dimension of threads
    
    int plateNum;   //numbers of plates
    int* plateIn;   //number of input flows plates
    float plateVol; //plates volume
    int plateInit;  //initial state of plates
    //
    int potNum; //numbers of potential parameters
    bool platesInsertion;   //insertion molecules into plates
} config;

extern struct molecules{
    int atomNum;
    char** aName;
    int* aType;
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

typedef struct{
    int* molNum;    //number of molecules in plate  // [plate number]
    int** molNumType;    //numbers of molecules of each type
    //[plate number][molecule type]
    
    int* state; //1 -- liquid 2 -- vapor    //
    float* liqVol;    //volume of liquid box    //[plate number]
    float* vapVol;    //volume of vapor box     //[plate number]
    float* temp;    //plate temperature
    
    float* refEnergy;   //reference energy of plate //[plate number]
    
    float* liqRcut; //cut radius for phases
    float* vapRcut;
    //float* liqEnergy;   //current energy of liquid phase
    
    float** xm; //[plate number][molecule number]
    float** ym;
    float** zm;
    int** mType;  //type of molecules   //[plate number][molecule number]
    int** gpuIndex;    //internal GPU index of molecule //[plate number] [molecule number]
    int* nLiq;  //number of molecules in liquid phase (per plate) //[platenumber]
    int* nVap;  //numbers of molecules in vapor phase (per plate) //[platenumber]
    
    
    int** liqList;   //list of liquid molecules per plate
        //[platenumber] [molecule index]
    int** vapList;  //list of vapor molecules per plate
        //[plate number] [molecule index]
    
    int** phaseType; //type of molecule phase [plate][molecule]
    
    float*** xa;    //atoms     //[plate number] [molecule id] [atom index]
    float*** ya;
    float*** za;
    
    int** type; //type of molecules //NOT USED
    
    int* plateDevice;    //number of device for plate
    int* devicePlates;  //numbers of plates per device
    int** platesPerDevice; //array of plates number per device [device] [plate]
    
} hDoubleBox;

//GPU structures
typedef struct{
    float* xm;
    float* ym;
    float* zm;
    int* molNum;    //numbers of molecules
    int* typeMolNum;    //numbers of each type of molecule
    int* mType; //type of molecule
    //
    float* xa;
    float* ya;
    float* za;
    int* aType; //type of atom

    int* fMol;  //first molecule of flow
    int* fAtom; //first atom of molecule
    
    float* boxLen; //box length
    float* boxVol;  //box volume
    
    //calculate total number of molecules and atoms
    int tMol;
    int tAtom;
    
    //flow energy
    float* virial;  //energy and virial of simulation cell
    float* energy;
    float* pressure;
    //float* LJEnergy;
    
    //enegry and virial per molecule
    float* mVirial;
    float* mEnergy;
    
    float* mEnergyT;    //for total energy
    float* mVirialT;
    
    float* oldEnergy;   //old and new viral/energy
    float* oldVirial;
    float* newEnergy;
    float* newVirial;
    
    
    float* transMaxMove;
    int* curMol;
    //flow prop
    int* accept;
    int* reject;
    int* tAccept;   //total accepted rejected per block
    int* tReject;
    
    float* eqEnergy;    //current energy
    float* eqPressure;  //current pressure
    
    float* eqBlockEnergy;   //energy of block samples
    float* eqBlockPressure;
    
    float* avEnergy;   //average energy of block samples
    float* avPressure;
    
    float* energyCorr;  //corrections
    float* pressureCorr;
} gSingleBox;


typedef struct{
    int* pltNum;    //plates number for each devcice
    int* fPlateNdx;   //first plate in each device
    int* pltList;   //list of plates in devices
    

    int* molNum;    //total numbers of molecules per plate
//    int* molNumType;    //total number of molecules by type per plate
    int* fMolNdx;   //first molecules index 
    
    float* xm;  //molecules coordinats
    float* ym;
    float* zm;
    
    float* xa;  //atoms coordinates
    float* ya;
    float* za;
    
    int* mType; //molecule type
    int* nLiq;  //number of liquid molecules
    int* nVap;  //number of vapor molecules
    
    int* fMolOnPlate;   //index of first molecule on plate [plates]
    int* fAtomOfMol;
//    int* fAtomNdx;    //index of first atopm of molecule [moleculas]
    
    int* aType; //atomtype
    
    int* liqList;   //lists of molecules
    int* vapList;
    int* phaseType;     //type of phase
    
    int* eqStep;    //nubers of step of equlibration on plate
    
    float* liqVol;  //volume of phases
    float* vapVol;
    
    float* liqRcut; //cut radius of phases
    float* vapRcut;
    
    float* liqEn;   //energy
    float* vapEn;
    float* refEn;
    
    float* liqVir;  //virial
    float* vapVir;
    
    float* tempLiqEn;  //temp array for enegry
    float* tempLiqVir;
    float* tempVapEn;   //temp vapor energy
    float* tempVapVir;
    
    float* tempXm;  //coordinates
    float* tempYm;
    float* tempZm;
    
    cudaStream_t stream;    //streams
    
    float* temp;    //current plate temperature
    
    float* maxLiqTrans;    //maximum liquid transition
    float* maxVapTrans;     //maximum vapor transition
    
    float* maxVolChange;    //maximumum volume change
    
    int* accLiqTrans;    //accept move in liquid phase
    int* rejLiqTrans;
    
    int* accVapTrans;   //accept move in vapor phase
    int* rejVapTrans;
    
    int* accVolChange;   //accept volume change
    int* rejVolChange;
    
    int* accLiq2Vap;    //accept liquid to vapor phase
    int* rejLiq2Vap;
    int* accVap2Liq;
    int* rejVap2Liq;
    
    //properties
    float* sumLiqEn;    //energy
    float* sumVapEn;
    float* sumLiqMolEn; //energy per molecules
    float* sumVapMolEn;
    float* sumLiqMol;   //moleecule numbers
    float* sumVapMol;
    float* sumLiqConc;  //concentration
    float* sumVapConc;
    float* sumLiqPress; //pressure
    float* sumVapPress;
    float* sumLiqMassDens;  //density
    float* sumVapMassDens;
    
    //
    float* blockLiqEn;  //energy
    float* blockVapEn;
    float* blockLiqMolEn;   //energy per molecule
    float* blockVapMolEn;
    float* blockLiqMol; //molecule number
    float* blockVapMol;
    float* blockLiqConc;    //concentration
    float* blockVapConc;
    float* blockLiqPress;   //pressure
    float* blockVapPress;
    float* blockLiqMassDens;    //density
    float* blockVapMassDens;
    
    float* avLiqEn; //energy
    float* avVapEn;
    float* avLiqMol;    //molecule numbers
    float* avVapMol;
    float* avLiqConc;   //concentration
    float* avVapConc;
    float* avLiqPress;  //pressure
    float* avVapPress;
    float* avLiqMassDens;   //density
    float* avVapMassDens;
    
} gDoublebox;

typedef struct{ //topology
    int* aNum;  //numbers of atom in molecule
    int* aType;    //type of atom in molecule
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
    int* subNum;
    int flowNum;
    int* potNum;
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
int data_to_device(gSingleBox &gBox, singleBox* &inputData, gOptions* &gConf, options &config, gMolecula* &gTop, potentialParam* Param, molecules* initMol, gSingleBox &hostData, gMolecula &hostTop, gOptions &hostConf, int deviceCount, gDoublebox* &gDBox);
int rcut(gSingleBox &hostData, options config, gMolecula hostTop, molecules* &initMol, gOptions &hConf);


__global__ void single_calc(int yDim,gOptions gConf, gMolecula gTop, gSingleBox gBox);
__device__ int single_calc_totenergy(int yDim, gOptions gConf, gMolecula gTop, gSingleBox &gBox);

__device__ int inter_potential(int a, int b, gOptions gConf, gMolecula gTop, gSingleBox &gBox, float &En, float &Vir);
__device__ int intra_potential(int a, gOptions gConf, gMolecula gTop, gSingleBox &gBox);

int data_from_device(gSingleBox &gData, gSingleBox &hData, options config);
int write_singlebox_log(FILE* logFile, gSingleBox &hData);

int plates_initial_state(options &config, hDoubleBox &doubleBox, gSingleBox &hostData, molecules* initMol, int deviceCount);
int double_box_init_allocate(options &config, hDoubleBox &doubleBox, int deviceCount);

int double_box_host_to_device(options &config, hDoubleBox &doubleBox, gDoublebox* &gDBox, gDoublebox &hDBox, gSingleBox &hostData, molecules* initMol, int deviceCount);

int double_equilibration(gDoublebox* gDBox, hDoubleBox doubleBox, gOptions* gConf, gMolecula* gTop);

//__global__ void 


#endif
//
