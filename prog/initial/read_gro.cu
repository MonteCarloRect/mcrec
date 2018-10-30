#include <stdio.h>
#include <unistd.h>
#include <ctype.h>
#include "../mcrec.h"
#include "../global.h"

int read_init_gro(options config, molecules* &initial){
    FILE* fId;
    char tempString[BUFFER];
    char tempString2[BUFFER];
    char tempString3[5];
    for(int i=0;i<config.subNum;i++){
        printf("subfile %s ",config.subFile[i]);
        if(access(config.subFile[i], F_OK ) != -1){
            fId=fopen(config.subFile[i],"r");
            fgets(tempString,BUFFER,fId);
//            fscanf(fId,"%d",&initial[i].atomNum);
            fgets(tempString,BUFFER,fId);
            initial[i].atomNum=atoi(tempString);
            printf("atom number %d\n", initial[i].atomNum);
            initial[i].aName=(char**) malloc (initial[i].atomNum *sizeof(char*));
            initial[i].aType=(unsigned int*) malloc(initial[i].atomNum *sizeof(int));
            for(int j=0;j<initial[i].atomNum;j++){
                initial[i].aName[j]=(char*)malloc (5*sizeof(char));
            }
            initial[i].x=(float*) malloc (initial[i].atomNum * sizeof(float));
            initial[i].y=(float*) malloc (initial[i].atomNum * sizeof(float));
            initial[i].z=(float*) malloc (initial[i].atomNum * sizeof(float));
            initial[i].vx=(float*) malloc (initial[i].atomNum * sizeof(float));
            initial[i].vy=(float*) malloc (initial[i].atomNum * sizeof(float));
            initial[i].vz=(float*) malloc (initial[i].atomNum * sizeof(float));
            for(int j=0;j<initial[i].atomNum;j++){
                fgets(tempString, BUFFER, fId);
                strncpy(tempString3,tempString+10,5);
                text_left(tempString3,initial[i].aName[j]);
                initial[i].x[j]=atof(strncpy(tempString2,tempString+20,8));
                initial[i].y[j]=atof(strncpy(tempString2,tempString+28,8));
                initial[i].z[j]=atof(strncpy(tempString2,tempString+36,8));
//                printf("x |%s|%f|%f|%f|\n",tempString2,initial[i].x[j],initial[i].y[j],initial[i].z[j]);
            }
            fclose(fId);
        }
        else{
            printf("file %s does not exist\n",config.subFile[i]);
            return 1;
        }
    }
    return 0;
}

int text_left(char* in, char* &out){
    int id=0;
    for(int i=0;i<5;i++){
        if(!isspace(in[i])){
            out[id]=in[i];
            id++;
        }
    }
    if(id<5){
        while(id<5){
            out[id]=' ';
            id++;
        }
    }
return 1;
}
