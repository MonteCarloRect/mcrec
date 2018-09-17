#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "../mcrec.h"

int read_top(potentialParam* &allParams,int &lines){
FILE* fileId;
char* fcontent = NULL;
char tempString[BUFFER];
char tempString2[BUFFER];
int i;
int curRead;
int firstSpase;
int id;
int wordNum;
int atomLines;
char* pEnd;
potentialParam tempParams[BUFFER];

    fileId=fopen("data.top","r");
//    fgets(tempString,BUFFER,fileId);
//    printf("%d %s",i,tempString);
    curRead=0;
    atomLines=0;
    while(fgets(tempString,BUFFER,fileId) != NULL) {
        i++;
//        printf("%s",tempString);
        if(strstr(tempString, "[") != NULL){
            curRead=0;  //set not to read
            if((strstr(tempString, "atomtypes") != NULL)){
                curRead=1;
            }
        }
        else{
            switch(curRead){
            case 1: //
                printf("%s",tempString);
               //while not a space
               firstSpase=1;
               id=0;
               wordNum=0;
               
            for(int charNum=0;charNum<BUFFER;charNum++){
                    if(!isspace(tempString[charNum])){
                        tempString2[id]=tempString[charNum];
                        firstSpase=0;   //
                        id++;   //
//                        printf("char %c id %d fs %d\n",tempString[charNum],id,firstSpase);
                    }
                    else{
                        if(firstSpase!=1){
                            firstSpase=1;
                            if(wordNum==0){
                                //copy first 5 char
                                strcpy(tempParams[atomLines].aName,"11111");    //fill 
                                for(int f5=0;f5<5;f5++){
                                    tempParams[atomLines].aName[f5]=tempString2[f5];
                                }
                                printf("aname %s %d\n",tempParams[atomLines].aName,id);
                            }
                            if(wordNum==1){
                                tempParams[atomLines].mass=strtof(tempString2,&pEnd);
                                printf("mass %f\n",tempParams[atomLines].mass);
                            }
                            if(wordNum==2){
                                tempParams[atomLines].charge=strtof(tempString2,&pEnd);
                                printf("charge %f\n",tempParams[atomLines].charge);
                            }
                            if(wordNum==4){
                                tempParams[atomLines].sigma=strtof(tempString2,&pEnd);
                                printf("sigma %f\n",tempParams[atomLines].sigma);
                            }
                            if(wordNum==5){
                                tempParams[atomLines].epsilon=strtof(tempString2,&pEnd);
                                printf("epsilon %f\n",tempParams[atomLines].epsilon);
                            }
                            id=0;
                            
                            //parse to type
                            wordNum++;
//                            printf("wornum %d\n",wordNum);
                        }
                        
                    }
                }
                atomLines++;
            break;
            }
        }
        
        
    }
    fclose(fileId);
    //creat array
    lines=atomLines;
    allParams=(potentialParam*)malloc(atomLines*sizeof(potentialParam));
    for(int i=0;i<atomLines;i++){
        memcpy(&allParams[i],&tempParams[i],sizeof(potentialParam));
    }
    
    return 0;
}

char* remove_space(char* input){
char *output;
int curLen;

curLen=strlen(input);
output=(char*)malloc(curLen*sizeof(char));
int j=0;
for(int i=0;i<curLen;i++){
//    output[i]=' ';
//    if(input[i] !=' '){
//        j++;
        output[j]=input[i];
//    }
}
//
return output;
}


