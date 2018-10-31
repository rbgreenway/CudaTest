#ifndef __CUDA_IMAGE_H__
#define __CUDA_IMAGE_H__

#include "vector_types.h"

#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>


#include "CudaUtility.h"

class CudaImage
{
public:

	CudaImage();
	~CudaImage();


	void ConvertGrayscaleToColor(uint8_t* color, uint16_t* gray, uint8_t* redMap, uint8_t* greenMap, uint8_t* blueMap,
		uint16_t width, uint16_t height, uint16_t maxGrayValue, uint16_t scaleLower, uint16_t scaleUpper);

	void CopyRoiToFullImage(uint16_t* full, uint16_t* roi, uint16_t fullW, uint16_t fullH,
		uint16_t  roiX, uint16_t roiY, uint16_t roiW, uint16_t roiH);

	void MaskImage(uint16_t* image, uint16_t* mask, uint16_t width, uint16_t height);

	void FlattenImage(uint16_t* image, float* Gc, float* Dc, uint16_t width, uint16_t height);

	void CopyCudaArrayToD3D9Memory(uint8_t* pDest, uint8_t* pSource, uint16_t pitch, uint16_t width, uint16_t height);

	void ComputeHistogram_512(uint32_t* hist, const uint16_t* data, uint16_t width, uint16_t height, uint8_t maxValueBitWidth);

	void BuildHistogramImage_512(uint8_t* histImage, uint32_t* hist, uint16_t numBins, uint16_t width, uint16_t height, uint32_t maxBinCount);

	void CalcApertureSums(uint32_t* sumArray, uint16_t* image, uint16_t* mask, uint16_t width, uint16_t height);


	uint16_t* SetFullGrayscaleImage(uint16_t* grayImage, uint16_t imageWidth, uint16_t imageHeight);

	uint16_t* SetRoiGrayscaleImage(uint16_t* roiImage, uint16_t imageWidth, uint16_t imageHeight, uint16_t roiWidth,
		uint16_t roiHeight, uint16_t roiX, uint16_t roiY);

	uint16_t* SetMaskImage(uint16_t* maskImage, uint16_t maskWidth, uint16_t maskHeight, uint16_t maskRows, uint16_t maskCols);

	void SetColorMap(uint8_t* redMap, uint8_t* greenMap, uint8_t* blueMap, uint16_t maxPixelValue);

	uint8_t* ConvertGrayscaleToColor(uint16_t scaleLower, uint16_t scaleUpper);

	void ApplyMaskToImage();

	uint8_t* PipelineFullImage(uint16_t* grayImage, uint16_t imageWidth, uint16_t imageHeight, bool applyMask);

	uint8_t* PipelineRoiImage(uint16_t* roiImage, uint16_t imageWidth, uint16_t imageHeight,
		uint16_t roiWidth, uint16_t roiHeight, uint16_t roiX, uint16_t roiY, bool applyMask);


	void Init();

	void Shutdown();

	void GetHistogram_512Buckets(uint32_t* destHist, uint8_t maxValueBitWidth);

	void GetHistogramImage_512Buckets(uint8_t* histImage, uint16_t width, uint16_t height, uint32_t maxBinCount);

	void CalculateMaskApertureSums(uint32_t* sums);

	void SetFlatFieldCorrectionArrays(int type, float* Gc, float* Dc, int numElements);

	void FlattenImage(int type);

	uint64_t SumImage(uint16_t* grayImage, uint16_t width, uint16_t height);

	uint64_t SumLoadedGrayImage();

	void Test();



	uint16_t m_imageW;
	uint16_t m_imageH;
	uint16_t m_roiW;
	uint16_t m_roiH;
	uint16_t m_roiX;
	uint16_t m_roiY;
	uint16_t m_maskW;
	uint16_t m_maskH;
	uint16_t m_maskRows;
	uint16_t m_maskCols;
	uint16_t m_maxPixelValue;

	uint16_t * mp_d_grayImage;
	uint8_t  * mp_d_colorImage;
	uint16_t * mp_d_maskImage;
	uint16_t * mp_d_roiImage;

	uint32_t * mp_d_histogram;
	uint8_t *  mp_d_colorHistogramImage;
	uint32_t   m_max_histogramBinValue;

	uint8_t * mp_d_redMap;
	uint8_t * mp_d_greenMap;
	uint8_t * mp_d_blueMap;

	uint32_t * mp_d_maskApertureSums; // 1D array, holds the aperture pixel sums

	float    * mp_d_FFC_Fluor_Gc; // 1D binning-corrected array, that holds the "gain" value for each pixel for Fluorescent Flat Field Correction
	float	 * mp_d_FFC_Fluor_Dc; // binning-corrected "dark" array that pairs with the "gain" array above

	float    * mp_d_FFC_Lumi_Gc; // 1D binning-corrected array, that holds the "gain" value for each pixel for Luminescenct Flat Field Correction
	float	 * mp_d_FFC_Lumi_Dc; // binning-corrected "dark" array that pairs with the "gain" array above
	uint32_t   m_h_FFC_numElements; // this holds the size of the correction arrays (should be imageWidth * imageHeight)


	bool    m_colorMapSet;
	bool    m_maskSet;
};

#endif