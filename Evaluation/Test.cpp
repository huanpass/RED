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

extern std::vector< std::vector<double> > testInput;
extern std::vector< std::vector<int> > dTestInput;
extern std::vector< std::vector<double> > testInput_0;
extern std::vector< std::vector<int> > dTestInput_0;
// extern std::vector< std::vector<double> > testOutput;

extern std::vector< std::vector<double> > weight1;
// extern std::vector< std::vector<double> > weight2;

extern Technology techIH;
// extern Technology techHO;
extern Array *arrayIH;
extern Array *arrayIH_0;
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

extern int correct;		// # of correct prediction

/* Validation */
void Validate(int ispadding) {
	int numBatchReadSynapse;    // # of read synapses in a batch read operation (decide later)
	double outN1[param->nHide]; // Net input to the hidden layer [param->nHide]
	double a1[param->nHide];    // Net output of hidden layer [param->nHide] also the input of hidden layer to output layer
	int da1[param->nHide];  // Digitized net output of hidden layer [param->nHide] also the input of hidden layer to output layer
	
	double outN1_0[param->nHide_0]; // Net input to the hidden layer [param->nHide]
	double a1_0[param->nHide_0];    // Net output of hidden layer [param->nHide] also the input of hidden layer to output layer
	int da1_0[param->nHide_0];  // Digitized net output of hidden layer [param->nHide] also the input of hidden layer to output layer	
	// double outN2[param->nOutput];   // Net input to the output layer [param->nOutput]
	// double a2[param->nOutput];  // Net output of output layer [param->nOutput]
	// double tempMax;
	// int countNum;
	correct = 0;


	// double sumArrayReadEnergyHO = 0;    // Use a temporary variable here since OpenMP does not support reduction on class member
	// double sumNeuroSimReadEnergyHO = 0; // Use a temporary variable here since OpenMP does not support reduction on class member
	// double sumReadLatencyHO = 0;    // Use a temporary variable here since OpenMP does not support reduction on class member
	// double readVoltageHO = static_cast<eNVM*>(arrayHO->cell[0][0])->readVoltage;
	// double readPulseWidthHO = static_cast<eNVM*>(arrayHO->cell[0][0])->readPulseWidth;
	if(!ispadding){
	double sumArrayReadEnergyIH = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
	double sumNeuroSimReadEnergyIH = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
	double sumReadLatencyIH = 0;    // Use a temporary variable here since OpenMP does not support reduction on class member
	double readVoltageIH = static_cast<eNVM*>(arrayIH->cell[0][0])->readVoltage;
	double readPulseWidthIH = static_cast<eNVM*>(arrayIH->cell[0][0])->readPulseWidth;
#pragma omp parallel for private(outN1, a1, da1,outN1_0, a1_0, da1_0,/* outN2, a2, tempMax, countNum, */numBatchReadSynapse) reduction(+: correct, sumArrayReadEnergyIH, sumNeuroSimReadEnergyIH,/* sumArrayReadEnergyHO, sumNeuroSimReadEnergyHO, */sumReadLatencyIH/*, sumReadLatencyHO*/)
	for (int i = 0; i < param->numMnistTestImages; i++)
	{
		// Forward propagation
		/* First layer from input layer to the hidden layer */
		//std::fill_n(outN1, param->nHide, 0);
		//std::fill_n(a1, param->nHide, 0);
			for (int j=0; j<param->nHide; j++) {
				if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
					if (static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) {  // 1T1R
						sumArrayReadEnergyIH += arrayIH->wireGateCapRow * techIH.vdd * techIH.vdd * param->nInput; // All WLs open
					}
				} else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayIH->cell[0][0])) { // Digital eNVM
					// XXX: To be released
				}
				for (int n=0; n<param->numBitInput; n++) {
					double pSumMaxAlgorithm = pow(2, n) / (param->numInputLevel - 1) * arrayIH->arrayRowSize;   // Max algorithm partial weighted sum for the nth vector bit (if both max input value and max weight are 1)
					if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
						double Isum = 0;    // weighted sum current
						double IsumMax = 0; // Max weighted sum current
						double inputSum = 0;    // Weighted sum current of input vector * weight=1 column
						for (int k=0; k<param->nInput; k++) {
							if ((dTestInput[i][k]>>n) & 1) {    // if the nth bit of dTestInput[i][k] is 1
								Isum += arrayIH->ReadCell(j,k);
								inputSum += arrayIH->GetMaxCellReadCurrent(j,k);
								sumArrayReadEnergyIH += arrayIH->wireCapRow * readVoltageIH * readVoltageIH;   // Selected BLs (1T1R) or Selected WLs (cross-point)
							}
							IsumMax += arrayIH->GetMaxCellReadCurrent(j,k);
						}
						sumArrayReadEnergyIH += Isum * readVoltageIH * readPulseWidthIH;
						int outputDigits = 2 * CurrentToDigits(Isum, IsumMax) - CurrentToDigits(inputSum, IsumMax);
						outN1[j] += DigitsToAlgorithm(outputDigits, pSumMaxAlgorithm);
					} else {    // SRAM or digital eNVM
						// XXX: To be released
					}
				}
				a1[j] = sigmoid(outN1[j]);
				da1[j] = round_th(a1[j]*(param->numInputLevel-1), param->Hthreshold);
			}


			numBatchReadSynapse = (int)ceil((double)param->nHide/param->numColMuxed);
			#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables
			for (int j=0; j<param->nHide; j+=numBatchReadSynapse) {
				int numActiveRows = 0;  // Number of selected rows for NeuroSim
				for (int n=0; n<param->numBitInput; n++) {
					for (int k=0; k<param->nInput; k++) {
						if ((dTestInput[i][k]>>n) & 1) {    // if the nth bit of dTestInput[i][k] is 1
							numActiveRows++;
						}
					}
				}
				subArrayIH->activityRowRead = (double)numActiveRows/param->nInput/param->numBitInput;
				sumNeuroSimReadEnergyIH += NeuroSimSubArrayReadEnergy(subArrayIH);
				sumNeuroSimReadEnergyIH += NeuroSimNeuronReadEnergy(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
				sumReadLatencyIH += NeuroSimSubArrayReadLatency(subArrayIH);
				sumReadLatencyIH += NeuroSimNeuronReadLatency(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
			}
	}
	if (!param->useHardwareInTraining) {    // Calculate the classification latency and energy only for offline classification
		arrayIH->readEnergy += sumArrayReadEnergyIH;
		subArrayIH->readDynamicEnergy += sumNeuroSimReadEnergyIH;
		// arrayHO->readEnergy += sumArrayReadEnergyHO;
		// subArrayHO->readDynamicEnergy += sumNeuroSimReadEnergyHO;
		subArrayIH->readLatency += sumReadLatencyIH;
		// subArrayHO->readLatency += sumReadLatencyHO;
	}
    }

	else{
	double sumArrayReadEnergyIH = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
	double sumNeuroSimReadEnergyIH = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
	double sumReadLatencyIH = 0;    // Use a temporary variable here since OpenMP does not support reduction on class member
	double readVoltageIH = static_cast<eNVM*>(arrayIH_0->cell[0][0])->readVoltage;
	double readPulseWidthIH = static_cast<eNVM*>(arrayIH_0->cell[0][0])->readPulseWidth;
	#pragma omp parallel for private(outN1, a1, da1,outN1_0, a1_0, da1_0,/* outN2, a2, tempMax, countNum, */numBatchReadSynapse) reduction(+: correct, sumArrayReadEnergyIH, sumNeuroSimReadEnergyIH,/* sumArrayReadEnergyHO, sumNeuroSimReadEnergyHO, */sumReadLatencyIH/*, sumReadLatencyHO*/)		
	for (int i = 0; i < param->numMnistTestImages_0; i++)
	{
		// Forward propagation
		/* First layer from input layer to the hidden layer */
		//std::fill_n(outN1_0, param->nHide_0, 0);
		//std::fill_n(a1_0, param->nHide_0, 0);
			for (int j=0; j<param->nHide_0; j++) {
				if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH_0->cell[0][0])) {  // Analog eNVM
					if (static_cast<eNVM*>(arrayIH_0->cell[0][0])->cmosAccess) {  // 1T1R
						sumArrayReadEnergyIH += arrayIH_0->wireGateCapRow * techIH.vdd * techIH.vdd * param->nInput_0; // All WLs open
					}
				} else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayIH_0->cell[0][0])) { // Digital eNVM
					// XXX: To be released
				}
				for (int n=0; n<param->numBitInput; n++) {
					double pSumMaxAlgorithm = pow(2, n) / (param->numInputLevel - 1) * arrayIH_0->arrayRowSize;   // Max algorithm partial weighted sum for the nth vector bit (if both max input value and max weight are 1)
					if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH_0->cell[0][0])) {  // Analog eNVM
						double Isum = 0;    // weighted sum current
						double IsumMax = 0; // Max weighted sum current
						double inputSum = 0;    // Weighted sum current of input vector * weight=1 column
						for (int k=0; k<param->nInput_0; k++) {
							if ((dTestInput_0[i][k]>>n) & 1) {    // if the nth bit of dTestInput_0[i][k] is 1
								Isum += arrayIH_0->ReadCell(j,k);
								inputSum += arrayIH_0->GetMaxCellReadCurrent(j,k);
								sumArrayReadEnergyIH += arrayIH_0->wireCapRow * readVoltageIH * readVoltageIH;   // Selected BLs (1T1R) or Selected WLs (cross-point)
							}
							IsumMax += arrayIH_0->GetMaxCellReadCurrent(j,k);
						}
						sumArrayReadEnergyIH += Isum * readVoltageIH * readPulseWidthIH;
						int outputDigits = 2 * CurrentToDigits(Isum, IsumMax) - CurrentToDigits(inputSum, IsumMax);
						outN1_0[j] += DigitsToAlgorithm(outputDigits, pSumMaxAlgorithm);
					} else {    // SRAM or digital eNVM
						// XXX: To be released
					}
				}
				a1_0[j] = sigmoid(outN1_0[j]);
				da1_0[j] = round_th(a1_0[j]*(param->numInputLevel-1), param->Hthreshold);
			}


			numBatchReadSynapse = (int)ceil((double)param->nHide_0/param->numColMuxed);
			#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables
			for (int j=0; j<param->nHide_0; j+=numBatchReadSynapse) {
				int numActiveRows = 0;  // Number of selected rows for NeuroSim
				for (int n=0; n<param->numBitInput; n++) {
					for (int k=0; k<param->nInput_0; k++) {
						if ((dTestInput_0[i][k]>>n) & 1) {    // if the nth bit of dTestInput_0[i][k] is 1
							numActiveRows++;
						}
					}
				}
				subArrayIH->activityRowRead = (double)numActiveRows/param->nInput_0/param->numBitInput;
				sumNeuroSimReadEnergyIH += NeuroSimSubArrayReadEnergy(subArrayIH);
				sumNeuroSimReadEnergyIH += NeuroSimNeuronReadEnergy(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
				sumReadLatencyIH += NeuroSimSubArrayReadLatency(subArrayIH);
				sumReadLatencyIH += NeuroSimNeuronReadLatency(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
			}
	}
	if (!param->useHardwareInTraining) {    // Calculate the classification latency and energy only for offline classification
		arrayIH_0->readEnergy += sumArrayReadEnergyIH;
		subArrayIH->readDynamicEnergy += sumNeuroSimReadEnergyIH;
		// arrayHO->readEnergy += sumArrayReadEnergyHO;
		// subArrayHO->readDynamicEnergy += sumNeuroSimReadEnergyHO;
		subArrayIH->readLatency += sumReadLatencyIH;
		// subArrayHO->readLatency += sumReadLatencyHO;	
	}		
	}
}
