#include <stdio.h>
#include <unistd.h>
#include "../mcrec.h"
#include "../global.h"

int read_init_gro(options config, molecules* &initial){
    FILE* fId;
    char tempString[BUFFER];
    char tempString2[BUFFER];
    for(int i=0;i<config.subNum;i++){
        printf("subfile %s ",config.subFile[i]);
        if(access(config.subFile[i], F_OK ) != -1){
            fId=fopen(config.subFile[i],"r");
            fgets(tempString,BUFFER,fId);
//            fscanf(fId,"%d",&initial[i].atomNum);
            fgets(tempString,BUFFER,fId);
            initial[i].atomNum=atoi(tempString);
            printf("atom number %d\n", initial[i].atomNum);
            initial[i].x=(float*) malloc (initial[i].atomNum * sizeof(float));
            initial[i].y=(float*) malloc (initial[i].atomNum * sizeof(float));
            initial[i].z=(float*) malloc (initial[i].atomNum * sizeof(float));
            initial[i].vx=(float*) malloc (initial[i].atomNum * sizeof(float));
            initial[i].vy=(float*) malloc (initial[i].atomNum * sizeof(float));
            initial[i].vz=(float*) malloc (initial[i].atomNum * sizeof(float));
            for(int j=0;j<initial[i].atomNum;j++){
                fgets(tempString, BUFFER, fId);
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
