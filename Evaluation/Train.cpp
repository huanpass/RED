/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
*   
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*   
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer. 
*   
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*   
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <iostream>
#include <vector>
#include <random>
#include "formula.h"
#include "Param.h"
#include "Array.h"
#include "Mapping.h"
#include "NeuroSim.h"

extern Param *param;

extern std::vector< std::vector<double> > Input;
extern std::vector< std::vector<int> > dInput;
// extern std::vector< std::vector<double> > Output;

extern std::vector< std::vector<double> > weight1;
extern std::vector< std::vector<double> > weight1_0;
// extern std::vector< std::vector<double> > weight2;
extern std::vector< std::vector<double> > deltaWeight1;
// extern std::vector< std::vector<double> > deltaWeight2;

extern Technology techIH;
// extern Technology techHO;
extern Array *arrayIH;
// extern Array *arrayHO;
extern SubArray *subArrayIH;
// extern SubArray *subArrayHO;
extern Adder adderIH;
extern Mux muxIH;
extern RowDecoder muxDecoderIH;
extern DFF dffIH;
// extern Adder adderHO;
// extern Mux muxHO;
// extern RowDecoder muxDecoderHO;
// extern DFF dffHO;
//const char* WeightFileName = "weight.txt";

void Train(/*const int numTrain, const int epochs*/int ispadding) {
	int size_2 = 25;
	int channel = 512;
	int batchsize = 256;
	double *weight_temp = new double[size_2*channel*batchsize];
	/*
    FILE *fp_weight = fopen(WeightFileName, "r");	
	if (!fp_weight) {
		std::cout << WeightFileName << " cannot be found!\n";
		exit(-1);
	}
	for(int i=0;i<channel * batchsize * size_2;i++){
			fscanf(fp_weight,"%lf",&weight_temp[i]);
	}
	fclose(fp_weight);
	*/
	srand(3);
	for(int i = 0;i< channel * batchsize ;i++){
		for (int j = 0;j< size_2;j++){
			weight_temp[i*size_2+j] = (double)(rand() % 1000) / 1000;   // random number: 0 to 1
		}
	}
	if(!ispadding)//nInput = batchsize, nHide = size_2*channel
	{
	    for (int j = 0; j < param->nHide; j++) {
			int j_col = j%size_2;
			int j_row = j/size_2;
		    for (int k = 0; k < param->nInput; k++) {
                weight1[j][k] = weight_temp[(j_row*channel+k)*size_2+j_col];		
		    }
	    }		 
	}
	else//nInput_0 = channel, nHide_0 = size_2*batchsize
	{
	    for (int j = 0; j < param->nHide_0; j++) {
		    for (int k = 0; k < param->nInput_0; k++) {
			int k_col = k%size_2;
			int k_row = k/size_2;				  
				weight1_0[j][k] = weight_temp[(k_row*batchsize+j)*size_2+k_col];	
		    }
	    } 		
	}	
}



