//
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "../mcrec.h"

int read_options(options &config){
    //varaibles
    FILE* fileId;
    char enStr[BUFFER];
    int curChar;
    char* ucString; //uncomment string
    char* rString;  //right part of prase
    char* lString;  //left part of string
    float sumnum;
    
    //------------add read from file soon
    config.mixRule=LB;
    //read by names
    printf("%s Read simulation options %s\n",ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    fileId = fopen("data.mcr", "r");
    if(fileId == NULL){
        printf("%s Can't open file data.mcr %s", ANSI_COLOR_RED, ANSI_COLOR_RESET);
        exit(EXIT_FAILURE);
    }
    //FIRST READ
    while (fgets(enStr, sizeof(enStr), fileId) != NULL){
        //get comments
        ucString = strtok(enStr,"#");
//        printf("uncoment %s \n", ucString);
        //check first phrase
        if(memchr(ucString, '=', strlen(ucString)) != NULL){
            rString = strtok(ucString, "=");   //first token
            //printf(" r string %s \n", rString);
            lString = strtok(NULL, "=");
            //printf(" left string %s\n", lString);
            //get numbers of substance
            if(strstr(rString,"substance_number") != NULL){
                config.subNum = atoi(lString);
                printf("Substance numbers %d\n", config.subNum);
            }
            //get flow numbers
            if(strstr(rString,"input_flows") != NULL){
                config.flowNum = atoi(lString);
                printf("Numbers of input flows %d\n", config.flowNum);
            }
            //get plates numbers
            if(strstr(rString,"plates_number") != NULL){
                config.plateNum = atoi(lString);
                printf("Numbers of plates %d\n", config.plateNum);
            }
           
            //get plates volume
            if(strstr(rString,"plates_vol") != NULL){
                config.plateVol = atof(lString);
                printf("Plates volume %f\n", config.plateVol);
            }
            //get initial state
            if(strstr(rString,"plates_init") != NULL){
                lString = strtok(lString, " \t\n");
                //printf("plates_init %s", lString);
                if(strcmp(lString,"vak")==0){
                    config.plateInit = VAK;
                    printf("Inital states of plates %d\n", config.plateInit);
                }
            }
        }
//        if(rString) = 
    }
    fclose(fileId);
    //printf("first read done\n");
    //malloc arrays
    config.flowEns = (int *) malloc(config.flowNum * sizeof(int));
    config.flowT = (float *) malloc(config.flowNum * sizeof(float));
    config.flowN = (float *) malloc(config.flowNum * sizeof(float));
    config.flowP = (float *) malloc(config.flowNum * sizeof(float));
    config.flowIns = (int **) malloc(config.flowNum * sizeof(int*));
    
    config.flowX = (float **) malloc(config.flowNum * sizeof(float *));
    config.plateIn = (int*) malloc(config.flowNum * sizeof(int));
    
    for (int i = 0; i < config.flowNum; i++) {
        config.flowX[i] = (float *) malloc(config.subNum * sizeof(float));
        config.flowIns[i] = (int *) malloc(config.subNum * sizeof(int));
    }
    //SECOND READ
    
        fileId = fopen("data.mcr", "r");
    if(fileId == NULL){
        printf("%s Can't open file data.mcr %s", ANSI_COLOR_RED, ANSI_COLOR_RESET);
        exit(EXIT_FAILURE);
    }
    while (fgets(enStr, sizeof(enStr), fileId) != NULL){
        //get comments
        ucString = strtok(enStr,"#");
//        printf("uncoment %s \n", ucString);
        //check first phrase
        if(memchr(ucString, '=', strlen(ucString)) != NULL){
            rString = strtok(ucString, "=");   //first token
            //printf(" r string %s \n", rString);
            lString = strtok(NULL, "=");
            //printf(" left string %s\n", lString);
            //get substance files
            if(strstr(rString,"substance_files") != NULL){
                lString = strtok(lString, " \t");
                strcpy(config.subFile[0], lString);
                printf("Files |%s|\n", config.subFile[0]);
                for(int i=1; i < config.subNum; i++){
                    lString = strtok(NULL, " \t\n");
                    strcpy(config.subFile[i], lString);
                    printf("Files |%s|\n", config.subFile[i]);
                }
            }
            //get flows ensamble
            if(strstr(rString,"input_ensamble") != NULL){
                lString = strtok(lString, " \t\n");
                //printf("%s", lString);
                if(strcmp(lString,"nvt")==0){
                    config.flowEns[0] = NVT;
                    printf("Flow 0 ensamble %d\n", config.flowEns[0]);
                }
                for(int i=1; i < config.flowNum; i++){
                    lString = strtok(NULL, " \t");
                    if(strcmp(lString,"nvt")==0){
                        config.flowEns[i]=NVT;
                        printf("Flow %d ensamble %d\n", i, config.flowEns[i]);
                    }
                }
            }
            //get flows temperatures
            if(strstr(rString,"input_temperature") != NULL){
                lString = strtok(lString, " \t");
                config.flowT[0] = atof(lString);
                printf("Flow 0 temperature %f\n", config.flowT[0]);
                for(int i=1; i < config.flowNum; i++){
                    lString = strtok(NULL, " \t");
                    config.flowT[i] = atof(lString);
                    printf("Flow %d temperature %f\n", i, config.flowT[i]);
                }
            }
            //get flows densities
            if(strstr(rString,"input_density") != NULL){
                lString = strtok(lString, " \t");
                config.flowN[0] = atof(lString);
                printf("Flow 0 density %f\n", config.flowN[0]);
                for(int i=1; i < config.flowNum; i++){
                    lString = strtok(NULL, " \t");
                    config.flowN[i] = atof(lString);
                    printf("Flow %d density %f\n", i, config.flowN[i]);
                }
            }
            //get flows pressure
            if(strstr(rString,"input_pressure") != NULL){
                lString = strtok(lString, " \t");
                config.flowP[0] = atof(lString);
                printf("Flow 0 density %f\n", config.flowP[0]);
                for(int i=1; i < config.flowNum; i++){
                    lString = strtok(NULL, " \t");
                    config.flowP[i] = atof(lString);
                    printf("Flow %d density %f\n", i, config.flowP[i]);
                }
            }
            //get flows composition
            
            if(strstr(rString,"input_ins_mol") != NULL){
//                printf("start read x\n");
                lString = strtok(lString, " \t");
                config.flowIns[0][0] = atoi(lString);
                printf("Flow 0 substance 0 inserted %d molecules\n", config.flowIns[0][0]);
                for(int i = 0; i < config.flowNum; i++){
                    for(int j = 0; j < config.subNum; j++){
                        //printf("i %d j %d\n",i,j);
                        if(!((i == 0) && (j == 0))){
                            lString = strtok(NULL, " \t\n");
                            //printf("lstring %s\n", lString);
                            config.flowIns[i][j] = atoi(lString);
                            printf("Flow 0 substance 0 inserted %d molecules\n", config.flowIns[i][j]);
                        }
                    }
                }
                for(int i = 0; i < config.flowNum; i++){
                sumnum = 0;
                for(int j = 0; j < config.subNum; j++){
                    sumnum += config.flowIns[i][j];
                }
                for(int j = 0; j < config.subNum; j++){
                    config.flowX[i][j] = config.flowIns[i][j] / sumnum;
                    printf("Flow %d substance %d composition %f\n", i, j, config.flowX[i][j]);
                }
            }
            }
            //перечитать состав
            
            
            //get flows numbers of inserted molecules
//            if(strstr(rString,"input_ins_mol") != NULL){
//                lString = strtok(lString, " \t");
//                config.flowIns[0] = atoi(lString);
//                printf("Flow 0 molecules inserted %d\n", config.flowIns[0]);
//                for(int i = 1; i < config.flowNum; i++){
//                    lString = strtok(NULL, " \t\n");
//                    config.flowIns[i] = atoi(lString);
//                    printf("Flow %d molecules inserted %d\n", i, config.flowIns[i]);
//                }
//            }
             //get input plate
            //printf("plates in %s\n",rString);
            if(strstr(rString,"plates_ins_number") != NULL){
                config.plateIn[0] = atoi(lString);
                printf("Flow  0 Number of plate %d\n", config.plateIn[0]);
                for(int i = 1; i < config.flowNum; i++){
                    lString = strtok(NULL, " \t\n");
                    config.plateIn[i] = atoi(lString);
                    printf("Flow  %d Number of plate %d\n", i, config.plateIn[i]);
                }
            }
            if(strstr(rString,"plates_insertion") != NULL){
                if(strstr(lString,"true") != NULL){
                    config.platesInsertion = true;
                }
                if(strstr(lString,"false") != NULL){
                    config.platesInsertion = false;
                }
                printf("Molecules insertion: %d\n", config.platesInsertion);
            }
            
        }
//        if(rString) = 
    }
    fclose(fileId);
    
//    //
//    fileId = fopen("data.mcr", "r");
//    fscanf(fileId, "%d", &config.subNum);
//    printf("Substance number: %d\n", config.subNum);
//    for (int i = 0; i < config.subNum; i++) {
//        fscanf(fileId, "%s", config.subFile[i]);
//    }
////read flow data
//    fscanf(fileId, "%d", &config.flowNum);

//    //read ensambles
//    for (int i = 0; i < config.flowNum; i++){
//        fscanf(fileId,"%s",&enStr);
////        printf("test|%s|\n",enStr);
//        if(strcmp(enStr,"nvt")==0){
//            config.flowEns[i]=NVT;
//            printf("ensamble %d\n",config.flowEns[i]);
//        }
//    }
//    for (int i = 0; i < config.flowNum; i++) {
//        fscanf(fileId, "%f", &config.flowT[i]);
//    }
//    for (int i = 0; i < config.flowNum; i++) {
//        fscanf(fileId, "%f", &config.flowN[i]);
//    }
//    for (int i = 0; i < config.flowNum; i++) {
//        fscanf(fileId, "%d", &config.flowIns[i]);
//    }
//    config.flowX = (float **) malloc(config.flowNum * sizeof(float *));
//    for (int i = 0; i < config.flowNum; i++) {
//        config.flowX[i] = (float *) malloc(config.subNum * sizeof(float));
//    }
//    for (int i = 0; i < config.flowNum; i++) {
//        for (int j = 0; j < config.subNum; j++) {
//            fscanf(fileId, "%f", &config.flowX[i][j]);
//        }
//    }
//    fclose(fileId);
    return 1;
}
