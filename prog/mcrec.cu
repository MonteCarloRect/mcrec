#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SUBNUMMAX 10
#define BUFFER 255
#define INSTIMES 10
#define MAXMOL 5000


int find_maximum(int a[], int n) {
int c, max, index;
	max = a[0];
	index = 0;
	for (c = 1; c < n; c++) {
		if (a[c] > max) {
			index = c;
		max = a[c];
	}
}
return a[index];
}

int get_integer(char a[], int begin, int length){
char b[length];
	for(int i=0;i<length;i++){
		b[i]=a[begin+i];
	}
	return atoi(b);
}

void get_string(char a[], char out[], int begin, int length){
	for(int i=0;i<length;i++){
		out[i]=a[begin+i];
	}
}

float get_float(char a[], int begin, int length){
char b[length];
	for(int i=0;i<length;i++){
		b[i]=a[begin+i];
	}
	return atof(b);
}

int main (int argc, char * argv[]){
	//----------------VAR
	int deviceCount;
	cudaDeviceProp temppd;	//temp varaible
	cudaDeviceProp* pd;	//array of device properties
	cudaError_t currentError;	//current error
	FILE* fileId;	//input file ID
	FILE* file2Id;	//input file ID

	//substance 
	int subNum;	//substance number
	char subFile[SUBNUMMAX][BUFFER];	//substance filenames
	int subAtomMax;	//maximum atom number in substances
	int* subAtomNum;	//atom numbers in molecules
	char** subName;	//substane residual name
	char*** subAtomName;	//attom names
	float3** subAtomCoord;	//atom position in molecule
	float3** subAtomVel;	//atom velocity
	

	//input flow
	int flowNum;	//flow numbers
	float** flowX;	//flow mole fractions
	int** flowNiIns;	//flow number of inserted molecules per cycle
	float* flowT;	//flow temperatures
	float* flowN;	//flow number density
	int* flowIns;	//inserting molecules per
	
	//temp varaibles
	char tempString[BUFFER],tempString2[BUFFER];
	int tempInt,tempInt2;
	float tempFloat;
	char* tempString3,tempString4;


	
	//init varaibles
	int* initMolNum;
	float3** initMolCoord_c;
	float3*** inttAtomCoord_c;
	int latice;
	int** initMolType_c;
	int** initBoxType_c;
	float3* tempCell;	//initial coordinates of 
	int* tempBusy;	//
	float Lbox;	//liquid cell length
	float Vbox;	//vapor cell length
	int id,id2;	//id of initial cell
	float liqFrac;
	int** initMolInsV;
	int** initMolInsL;
	int** initMolLiqList;	//list of molecules in liquid phase at plate
	int** initMolVapList;	//list of molecules in vapor phase at plate
	int* initMolLiqNum;	//numbers of molecules in liquid phase at plate
	int* initMolVapNum;	//number of molecules in vapor phase at plate


	//functions
	int find_maximum(int a[], int n);
	int get_integer(char a[], int n, int begin, int length);
	void get_string(char a[], char out[], int begin, int length);
	float get_float(char a[], int begin, int length);
	
	//random generator
	srand(time(NULL));
	//----------------GET DEVICE PROPERTIES
	currentError=cudaGetDeviceCount(&deviceCount);
	if (currentError!=cudaSuccess){
		fprintf(stderr,"Cannot get CUDA device count: %s\n", cudaGetErrorString(currentError));
		return 1;
	}
	if (!deviceCount){
		fprintf(stderr, "No CUDA devices found\n");
		return 1;
	}
	pd=(cudaDeviceProp*) malloc(deviceCount*sizeof(cudaDeviceProp));
	for (int i=0;i<deviceCount;i++){
		cudaGetDeviceProperties(&temppd,i);
//		priflowNlowT("Device name %s \n", pd.name);
		pd[i]=temppd;
		printf("Device name %s \n", pd[i].name);
		printf("Max Threads Dim: %d %d %d \n", pd[i].maxThreadsDim[0],pd[i].maxThreadsDim[1],pd[i].maxThreadsDim[2]);
		printf("Max Grid Size: %d %d %d \n", pd[i].maxGridSize[0], pd[i].maxGridSize[1], pd[i].maxGridSize[2]);
	}
	
	//------------------READ INPUT DATA
	fileId=fopen("data.mcr","r");
		fscanf(fileId,"%d",&subNum);
//read molecules data
		printf("Substance number: %d\n", subNum);
		for(int i=0;i<subNum;i++){
			fscanf(fileId,"%s",subFile[i]);
		}
//read flow data
		fscanf(fileId,"%d",&flowNum);
		flowT=(float*) malloc(flowNum*sizeof(float));
		flowN=(float*) malloc(flowNum*sizeof(float));
		flowIns=(int*) malloc(flowNum*sizeof(int));
		for(int i=0;i<flowNum;i++){
			fscanf(fileId,"%f",&flowT[i]);
		}
		for(int i=0;i<flowNum;i++){
			fscanf(fileId,"%f",&flowN[i]);
		}
		for(int i=0;i<flowNum;i++){
			fscanf(fileId,"%d",&flowIns[i]);
		}
		flowX=(float**) malloc(flowNum*sizeof(float*));
		for(int i=0;i<flowNum;i++){
			flowX[i]=(float*)malloc(subNum*sizeof(float));
		}
		flowNiIns=(int**) malloc(flowNum*sizeof(int*));
		for(int i=0; i<flowNum;i++){
			flowNiIns[i]=(int*) malloc(subNum*sizeof(int));
		}
		for(int i=0;i<flowNum;i++){
			for(int j=0;j<subNum;j++){
			fscanf(fileId,"%f", &flowX[i][j]);
			}
		}
		//calculate numbers of initial molecules
		for(int i=0;i<flowNum;i++){
			tempInt=0;
			for(int j=0;j<subNum-1;j++){
				flowNiIns[i][j]=floor(flowX[i][j]*flowIns[i]);
				tempInt+=flowNiIns[i][j];
			}
			flowNiIns[i][subNum-1]=flowIns[i]-tempInt;
		}
	fclose(fileId);
//--------------------READ GRO FILES
//read molecules structure 
	subAtomNum=(int*)malloc(subNum*sizeof(int));
	for(int i=0;i<subNum;i++){
		fileId=fopen(subFile[i],"r");
		 if (fileId == NULL){
			printf("Error opening file %s\n", subFile[i]);
			return 1;
		}
		fgets(tempString,BUFFER,fileId);
		fscanf(fileId,"%d",&subAtomNum[i]);
		fclose(fileId);
//			fscanf(file2Id,"%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f", )
	}
	//allocate gro varaibles
	subAtomMax=find_maximum(subAtomNum,subNum);
	printf("Maximum atoms numbers %d\n",subAtomMax);
	subName=(char**)malloc(subNum*sizeof(char*));	//allocate molecule names
	for(int i=0;i<subNum;i++){
		subName[i]=(char*)malloc(5*sizeof(char));
	}
	subAtomCoord=(float3**)malloc(subNum*sizeof(float3*));	//allocate coordinates
	for(int i=0;i<subNum;i++){
		subAtomCoord[i]=(float3*)malloc(subAtomMax*sizeof(float3));
	}
	subAtomVel=(float3**)malloc(subNum*sizeof(float3*));	//allocate velocities
	for(int i=0;i<subNum;i++){
		subAtomVel[i]=(float3*)malloc(subAtomMax*sizeof(float3));
	}
	subAtomName=(char***)malloc(subNum*sizeof(char**));	//allocate atoma names
	for(int i=0;i<subNum;i++){
		subAtomName[i]=(char**)malloc(subAtomMax*sizeof(char*));
		for(int j=0;j<subAtomMax;j++){
			subAtomName[i][j]=(char*)malloc(5*sizeof(char));
		}
	}
	for(int i=0;i<subNum;i++){
		fileId=fopen(subFile[i],"r");
		fgets(tempString,BUFFER,fileId);	//skip 2 lines
		fgets(tempString,BUFFER,fileId);
		for(int j=0;j<subAtomNum[i];j++){
			fgets(tempString2,BUFFER,fileId);	//read string to buffer
			get_string(tempString2,subName[i],5,5);	//parse string
			printf("sub %s\n",subName[i]);
			get_string(tempString2,subAtomName[i][j],10,5);
			printf("atom %s\n",subAtomName[i][j]);
			subAtomCoord[i][j].x=get_float(tempString2,20,8);
			subAtomCoord[i][j].y=get_float(tempString2,28,8);
			subAtomCoord[i][j].z=get_float(tempString2,36,8);
			printf("x %f y %f z %f  \n",subAtomCoord[i][j].x,subAtomCoord[i][j].y,subAtomCoord[i][j].z);
		}
		fclose(fileId);
	}
//---------------------READ TOPOLOGY



//------------Print initial data
	//print flow properties
	printf("#\tT,K\tn,mol/l\tinsert molecules\n");
	for(int i=0;i<flowNum;i++){
		printf("%d\t%f\t%f\t%d\n",i,flowT[i],flowN[i],flowIns[i]);
	}
	//molecule properties print
	printf("#\t molecule name\n");
	for(int i=0; i<subNum;i++){
		printf("%d\t%s\n",i,subFile[i]);
	}

	//------------------INITIAL SIMULATION
	initMolNum=(int*)malloc(flowNum*sizeof(int));
	initMolCoord_c=(float3**)malloc(flowNum*sizeof(float3*));
	initMolType_c=(int**)malloc(flowNum*sizeof(int*));
	initMolInsV=(int**)malloc(flowNum*sizeof(int*));
	initMolInsL=(int**)malloc(flowNum*sizeof(int*));
	initMolLiqList=(int**)malloc(flowNum*sizeof(int*));
	initMolVapList=(int**)malloc(flowNum*sizeof(int*));
	initMolLiqNum=(int*)malloc(flowNum*sizeof(int));
	initMolVapNum=(int*)malloc(flowNum*sizeof(int));
	//allocate initial arrays
	for(int i=0;i<flowNum;i++){
		initMolNum[i]=flowIns[i]*INSTIMES;	//numbers of molecules in initial plates 10 times ladger input flow
		initMolCoord_c[i]=(float3*)malloc(initMolNum[i]*sizeof(float3));	//allocate atom coords
		initMolType_c[i]=(int*)malloc(initMolNum[i]*sizeof(int));
		initMolInsV[i]=(int*)malloc(subNum*sizeof(int));
		initMolInsL[i]=(int*)malloc(subNum*sizeof(int));
		initMolLiqList[i]=(int*)malloc(initMolNum[i]*sizeof(int));
		initMolVapList[i]=(int*)malloc(initMolNum[i]*sizeof(int));
		for(int j=0;j<initMolNum[i];j++){
			initMolType_c[i][j]=-1;	//initial free space
		}
	}
	for (int i=0;i<flowNum;i++){
		//Insert in liquid box
		liqFrac=0.8;
		for(int j=0;j<subNum;j++){	//get liquid molecules per cell
			initMolInsL[i][j]=ceil(flowNiIns[i][j]*INSTIMES*liqFrac);
			initMolInsV[i][j]=flowNiIns[i][j]*INSTIMES-initMolInsL[i][j];
		}
		latice=ceil(pow(initMolNum[i]*liqFrac,1.0/3.0))+1;	//
		//insert in liquid
		printf("numbers %d\n",latice);
		printf("nmol %d \n",initMolNum[i]);
		tempCell=(float3*)malloc(latice*latice*latice*sizeof(float3));	//temporally cell array
		tempBusy=(int*)malloc(latice*latice*latice*sizeof(int));
		Lbox=10.0;	//написать выбор размеров начальных боксов
		id=0;
		for(int j=0;j<latice;j++){
			for(int k=0;k<latice;k++){
				for(int l=0;l<latice;l++){
					tempCell[id].x=(float)j*Lbox/(float)latice;
					tempCell[id].y=(float)k*Lbox/(float)latice;
					tempCell[id].z=(float)l*Lbox/(float)latice;
					tempBusy[id]=-1;
					id++;
				}
			}
		}
		id=0;
		for(int j=0;j<subNum;j++){
			tempInt=0;
			while(tempInt<initMolInsL[i][j]){
				tempInt2=rand() % (latice*latice*latice);
//				printf("random %d busy %d x %f \n",tempInt2,tempBusy[tempInt2],tempCell[tempInt2].x);
//				tempInt++;
				if(tempBusy[tempInt2]==-1){
//					initMolType_c[i]
					initMolType_c[i][id]=j;
					initMolCoord_c[i][id]=tempCell[tempInt2];
					initMolLiqList[i][id]=id;
					//liquid
					
					tempBusy[tempInt2]=1;
					tempInt++;
//					printf("%d sub %d id %d\n",id,j,initMolNum[i]);
//					printf(" x %f xl %f\n",initMolCoord_c[j][id].x,tempCell[tempInt2].x);
					id++;
				}
			}
		}
		free(tempCell);
		free(tempBusy);
		id2=id; //get the last element index
		//insert molecules in vapor=========================================
		latice=ceil(pow(initMolNum[i]*(1.0-liqFrac),1.0/3.0))+1;	//
		//insert in liquid
//		printf("numbers %d\n",latice);
//		printf("nmol %d \n",initMolNum[i]);
		tempCell=(float3*)malloc(latice*latice*latice*sizeof(float3));	//temporally cell array
		tempBusy=(int*)malloc(latice*latice*latice*sizeof(int));
		Lbox=40.0;	//написать выбор размеров начальных боксов
		id=0;
		for(int j=0;j<latice;j++){
			for(int k=0;k<latice;k++){
				for(int l=0;l<latice;l++){
					tempCell[id].x=(float)j*Lbox/(float)latice;
					tempCell[id].y=(float)k*Lbox/(float)latice;
					tempCell[id].z=(float)l*Lbox/(float)latice;
					tempBusy[id]=-1;
					id++;
				}
			}
		}
		initMolLiqNum[i]=id;	//numbers of molecules in liquid phase
		id=id2;
		id2=0;
		for(int j=0;j<subNum;j++){
			tempInt=0;
			while(tempInt<initMolInsV[i][j]){
				tempInt2=rand() % (latice*latice*latice);
//				printf("random %d busy %d x %f \n",tempInt2,tempBusy[tempInt2],tempCell[tempInt2].x);
//				tempInt++;
				if(tempBusy[tempInt2]==-1){
//					initMolType_c[i]
					initMolType_c[i][id]=j;	//set type of molecule
					initMolCoord_c[i][id]=tempCell[tempInt2];	//set coordinates from cubic latice
					initMolVapList[i][id2]=id;
					//vapor molecules list
						
					tempBusy[tempInt2]=1;
					tempInt++;
//					printf("%d sub %d id %d\n",id,j,initMolNum[i]);
//					printf(" x %f xl %f\n",initMolCoord_c[j][id].x,tempCell[tempInt2].x);
					id++;
					id2++;
				}
			}
		}
		initMolVapNum[i]=id2;
		free(tempCell);
		free(tempBusy);
	}
	//for initial calculates on first device
	
	
	
	

	//------------------FREE Arrays
//free
	
	free(flowT);
	free(flowN);
	free(pd);

	for(int i=0;i<flowNum;i++){
		free(flowX[i]);
		
		//initial arrays
		free(initMolCoord_c[i]);
	}
	free(flowX);

	free(initMolCoord_c);
	//free properties of devices


//	priflowNlowT("Device count %d \n", deviceCount);
//	cudaGetDeviceProperties(&pd,0);
//	priflowNlowT("Device name %s \n", pd.name);
//	priflowNlowT("Registers per block: %d \n",pd.regsPerBlock);
//	priflowNlowT("Max Threads Dim: %d %d %d \n", pd.maxThreadsDim[0],pd.maxThreadsDim[1],pd.maxThreadsDim[2]);
//	
	return 0;
}


