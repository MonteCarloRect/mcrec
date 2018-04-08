#include <stdio.h>
#include <stdlib.h>

#define SUBNUMMAX 10
#define BUFFER 255


int find_maximum(const int array[], int length) {
    int max = array[0];
    for (int i = 1; i < length; i++) {
        if (array[i] > max) {
            max = array[i];
        }
    }
    return max;
}

void get_string_section(const char string_input[], char string_output[], int begin, int length) {
    for (int i = 0; i < length; i++) {
        string_output[i] = string_input[begin + i];
    }
}

int get_integer_from_string(const char line[], int begin, int length) {
    char line_section[length];
    get_string_section(line, line_section, begin, length);
    return atoi(line_section);
}

float get_float(char a[], int begin, int length) {
    char b[length];
    for (int i = 0; i < length; i++) {
        b[i] = a[begin + i];
    }
    return atof(b);
}

int main(int argc, char *argv[]) {
    //----------------VAR
    int deviceCount;
    cudaDeviceProp temppd;    //temp varaible
    cudaDeviceProp *pd;    //array of device properties
    cudaError_t currentError;    //current error
    FILE *fileId;    //input file ID
    FILE *file2Id;    //input file ID

    //substance
    int subNum;    //substance number
    char subFile[SUBNUMMAX][BUFFER];    //substance filenames
    int subAtomMax;    //maximum atom number in substances
    int *subAtomNum;    //atom numbers in molecules
    char **subName;    //substane residual name
    char ***subAtomName;    //attom names
    float3 **subAtomCoord;    //atom position in molecule
    float3 **subAtomVel;    //atom velocity


    //input flow
    int flowNum;    //flow numbers
    float **flowX;    //flow mole fractions
    float *flowT;    //flow temperatures
    float *flowN;    //flow number density
    int *flowIns;    //inserting molecules per

    //temp varaibles
    char tempString[BUFFER], tempString2[BUFFER];
    int tempInt, tempInt2;
    float tempFloat;
    char *tempString3, tempString4;



    //init vaaraibles
    int *initMolNum;

    //functions
    int find_maximum(int a[], int n);
    int get_integer_from_string(char a[], int n, int begin, int length);
    void get_string_section(char a[], char out[], int begin, int length);
    float get_float(char a[], int begin, int length);

    //----------------GET DEVICE PROPERTIES
    currentError = cudaGetDeviceCount(&deviceCount);
    if (currentError != cudaSuccess) {
        fprintf(stderr, "Cannot get CUDA device count: %s\n", cudaGetErrorString(currentError));
        return 1;
    }
    if (!deviceCount) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    pd = (cudaDeviceProp *) malloc(deviceCount * sizeof(cudaDeviceProp));
    for (int i = 0; i < deviceCount; i++) {
        cudaGetDeviceProperties(&temppd, i);
//		priflowNlowT("Device name %s \n", pd.name);
        pd[i] = temppd;
        printf("Device name %s \n", pd[i].name);
        printf("Max Threads Dim: %d %d %d \n", pd[i].maxThreadsDim[0], pd[i].maxThreadsDim[1], pd[i].maxThreadsDim[2]);
        printf("Max Grid Size: %d %d %d \n", pd[i].maxGridSize[0], pd[i].maxGridSize[1], pd[i].maxGridSize[2]);
    }

    //------------------READ INPUT DATA
    fileId = fopen("data.mcr", "r");
    fscanf(fileId, "%d", &subNum);
//read molecules data
    printf("Substance number: %d\n", subNum);
    for (int i = 0; i < subNum; i++) {
        fscanf(fileId, "%s", subFile[i]);
    }
//read flow data
    fscanf(fileId, "%d", &flowNum);
    flowT = (float *) malloc(flowNum * sizeof(float));
    flowN = (float *) malloc(flowNum * sizeof(float));
    flowIns = (int *) malloc(flowNum * sizeof(int));
    for (int i = 0; i < flowNum; i++) {
        fscanf(fileId, "%f", &flowT[i]);
    }
    for (int i = 0; i < flowNum; i++) {
        fscanf(fileId, "%f", &flowN[i]);
    }
    for (int i = 0; i < flowNum; i++) {
        fscanf(fileId, "%d", &flowIns[i]);
    }
    flowX = (float **) malloc(flowNum * sizeof(float *));
    for (int i = 0; i < flowNum; i++) {
        flowX[i] = (float *) malloc(subNum * sizeof(float));
    }
    for (int i = 0; i < flowNum; i++) {
        for (int j = 0; j < subNum; j++) {
            fscanf(fileId, "%f", &flowX[i][j]);
        }
    }
    fclose(fileId);
//--------------------READ GRO FILES
//read molecules structure
    subAtomNum = (int *) malloc(subNum * sizeof(int));
    for (int i = 0; i < subNum; i++) {
        fileId = fopen(subFile[i], "r");
        if (fileId == NULL) {
            printf("Error opening file %s\n", subFile[i]);
            return 1;
        }
        fgets(tempString, BUFFER, fileId);
        fscanf(fileId, "%d", &subAtomNum[i]);
        fclose(fileId);
//			fscanf(file2Id,"%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f", )
    }
    //allocate gro varaibles
    subAtomMax = find_maximum(subAtomNum, subNum);
    printf("Maximum atoms numbers %d\n", subAtomMax);
    subName = (char **) malloc(subNum * sizeof(char *));    //allocate molecule names
    for (int i = 0; i < subNum; i++) {
        subName[i] = (char *) malloc(5 * sizeof(char));
    }
    subAtomCoord = (float3 **) malloc(subNum * sizeof(float3 * ));    //allocate coordinates
    for (int i = 0; i < subNum; i++) {
        subAtomCoord[i] = (float3 *) malloc(subAtomMax * sizeof(float3));
    }
    subAtomVel = (float3 **) malloc(subNum * sizeof(float3 * ));    //allocate velocities
    for (int i = 0; i < subNum; i++) {
        subAtomVel[i] = (float3 *) malloc(subAtomMax * sizeof(float3));
    }
    subAtomName = (char ***) malloc(subNum * sizeof(char **));    //allocate atoma names
    for (int i = 0; i < subNum; i++) {
        subAtomName[i] = (char **) malloc(subAtomMax * sizeof(char *));
        for (int j = 0; j < subAtomMax; j++) {
            subAtomName[i][j] = (char *) malloc(5 * sizeof(char));
        }
    }
    for (int i = 0; i < subNum; i++) {
        fileId = fopen(subFile[i], "r");
        fgets(tempString, BUFFER, fileId);    //skip 2 lines
        fgets(tempString, BUFFER, fileId);
        for (int j = 0; j < subAtomNum[i]; j++) {
            fgets(tempString2, BUFFER, fileId);    //read string to buffer
            get_string_section(tempString2, subName[i], 5, 5);    //parse string
            printf("sub %s\n", subName[i]);
            get_string_section(tempString2, subAtomName[i][j], 10, 5);
            printf("atom %s\n", subAtomName[i][j]);
            subAtomCoord[i][j].x = get_float(tempString2, 20, 8);
            subAtomCoord[i][j].y = get_float(tempString2, 28, 8);
            subAtomCoord[i][j].z = get_float(tempString2, 36, 8);
            printf("x %f y %f z %f  \n", subAtomCoord[i][j].x, subAtomCoord[i][j].y, subAtomCoord[i][j].z);
        }
        fclose(fileId);
    }
//---------------------READ TOPOLOGY



//------------Print initial data
    //print flow properties
    printf("#\tT,K\tn,mol/l\tinsert molecules\n");
    for (int i = 0; i < flowNum; i++) {
        printf("%d\t%f\t%f\t%d\n", i, flowT[i], flowN[i], flowIns[i]);
    }
    //molecule properties print
    printf("#\t molecule name\n");
    for (int i = 0; i < subNum; i++) {
        printf("%d\t%s\n", i, subFile[i]);
    }

    //------------------INITIAL SIMULATION
    initMolNum = (int *) malloc(flowNum * sizeof(int));
    for (int i = 0; i < flowNum; i++) {
        initMolNum[i] = flowIns[i] * 10;

    }




    //------------------FREE Arrays
    for (int i = 0; i < flowNum; i++) {
        free(flowX[i]);
    }
    free(flowX);
    free(flowT);
    free(flowN);
    free(pd);    //free properties of devices


//	priflowNlowT("Device count %d \n", deviceCount);
//	cudaGetDeviceProperties(&pd,0);
//	priflowNlowT("Device name %s \n", pd.name);
//	priflowNlowT("Registers per block: %d \n",pd.regsPerBlock);
//	priflowNlowT("Max Threads Dim: %d %d %d \n", pd.maxThreadsDim[0],pd.maxThreadsDim[1],pd.maxThreadsDim[2]);
//
    return 0;
}
