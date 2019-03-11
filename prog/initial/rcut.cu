#include <stdio.h>
#include "../mcrec.h"
#include "../global.h"

//calculate cut radius energy and pressure correction

int rcut(gSingleBox &hostData, options config, gMolecula hostTop, molecules* &initMol, gOptions &hConf){
    float sumE;
    float sumP;
    int id1;
    int id2;
    int aid;
    float rc;
    
    sumE = 0.0;
    sumP = 0.0;
    printf("E correction %f \n", sumE);
    for(int flowN = 0; flowN < config.flowNum; flowN++){  //flows
        for(int i = 0; i < config.subNum; i++){   //molecules
            for(int j = 0; j < config.subNum; j++){
                for(int a = 0; a < initMol[i].atomNum; a++){  //atoms
                    for(int b = 0; b < initMol[j].atomNum; b++){
                        id1 = config.flowNum*flowN + i;
                        id2 = config.flowNum*flowN + j;
                        aid = initMol[i].aType[a]* hConf.potNum[0] + initMol[j].aType[b];
                        rc = hostTop.sigma[aid] / hostData.boxLen[flowN] / 2.0;
                        rc = rc * rc * rc;
                        sumE += 2.0/9.0 * hostTop.epsi[aid] * PI * hostData.typeMolNum[id1] * hostData.typeMolNum[id2] / hostData.molNum[flowN] / hostData.boxVol[flowN] * hostTop.sigma[aid] * hostTop.sigma[aid] * hostTop.sigma[aid] * rc *rc *rc - 2.0 / 3.0 * PI * hostTop.epsi[aid] * PI * hostData.typeMolNum[id1] * hostData.typeMolNum[id2] / hostData.molNum[flowN] / hostData.boxVol[flowN] * hostTop.sigma[aid] * hostTop.sigma[aid] * hostTop.sigma[aid] * rc;
                        sumP += 8.0 / 9.0 * hostTop.epsi[aid] * PI * hostData.typeMolNum[id1] * hostData.typeMolNum[id2] / hostData.boxVol[flowN] / hostData.boxVol[flowN] * hostTop.sigma[aid] * hostTop.sigma[aid] * hostTop.sigma[aid] * rc *rc *rc - 4.0 / 3.0 * hostTop.epsi[aid] * PI * hostData.typeMolNum[id1] * hostData.typeMolNum[id2] / hostData.boxVol[flowN] / hostData.boxVol[flowN] * hostTop.sigma[aid] * hostTop.sigma[aid] * hostTop.sigma[aid] * rc;
                        //printf("id1 %d id2 %d aid %d rc %f \n", id1, id2, aid , rc);
                    }
                }
            }
        }
        sumP = sumP * 1.38064852e1;
        printf("E correction %f P correction %f \n", sumE, sumP);
        hostData.energyCorr[flowN] = sumE;
        hostData.pressureCorr[flowN] = sumP;
    }
    
    return 0;
}


