#include "stdafx.h"

#define DllExport  extern "C" __declspec( dllexport ) 

bool ready = false;

int range[4];

BOOL APIENTRY DllMain(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}


#pragma region CudaPlot


uchar4 UInt32_to_uchar4(uint32_t val)
{
	uchar4 result;

	result.x = (uint8_t)(val & 0x000000ff);
	result.y = (uint8_t)((val & 0x0000ff00) >> 8);
	result.z = (uint8_t)((val & 0x00ff0000) >> 16);
	result.w = (uint8_t)((val & 0xff000000) >> 24);

	return result;
}

DllExport bool InitCudaPlot(int chartRows, int chartCols, int chartArrayWidth, int chartArrayHeight, 
	int margin, int padding, int aggregateWidth, int aggregateHeight,
	uint32_t windowBkgColor,
	uint32_t chartBkgColor, uint32_t chartSelectedColor, uint32_t chartFrameColor, uint32_t chartAxisColor, uint32_t chartPlotColor,
	int xmin, int xmax, int ymin, int ymax,	int maxNumDataPoints, int numTraces, CudaChartArray** pp_chartArray)
{
	bool ready = true;

	int2 xRange = { xmin,xmax };
	int2 yRange = { ymin,ymax };

	uchar4 col1 = UInt32_to_uchar4(windowBkgColor);
	uchar4 col2 = UInt32_to_uchar4(chartBkgColor);
	uchar4 col3 = UInt32_to_uchar4(chartSelectedColor);
	uchar4 col4 = UInt32_to_uchar4(chartFrameColor);
	uchar4 col5 = UInt32_to_uchar4(chartAxisColor);
	uchar4 col6 = UInt32_to_uchar4(chartPlotColor);

	*pp_chartArray = new CudaChartArray(chartRows, chartCols, chartArrayWidth, chartArrayHeight, margin, padding,
									aggregateWidth, aggregateHeight,
									col1,col2,col3,col4,col5,col6, xRange, yRange, maxNumDataPoints, numTraces);

	return ready;
}



DllExport void Shutdown_ChartArray(CudaChartArray* p_chart_array)
{
	delete p_chart_array;
}


DllExport int GetMaxNumberOfTraces(CudaChartArray* pChartArray)
{
	return pChartArray->GetMaxNumberOfTraces();
}


DllExport int2 GetChartArrayPixelSize(CudaChartArray* pChartArray)
{	
	int2 size = pChartArray->GetChartArrayPixelSize();
	return size;
}


DllExport void Resize(CudaChartArray* pChartArray, int chartArrayWidth, int chartArrayHeight, int aggregateWidth, int aggregateHeight)
{
	pChartArray->Resize(chartArrayWidth, chartArrayHeight, aggregateWidth, aggregateHeight);
}


DllExport void SetSelected(CudaChartArray* pChartArray)
{
	int rows = pChartArray->m_rows;
	int cols = pChartArray->m_cols;
	int num = rows*cols;

	pChartArray->SetSelected();
}


DllExport void SetTraceColor(CudaChartArray* pChartArray, int traceNum, uint32_t color)
{
	uchar4 col1 = UInt32_to_uchar4(color);
	pChartArray->SetTraceColor(traceNum, col1);
}

DllExport void SetTraceVisibility(CudaChartArray* pChartArray, int traceNum, bool isVisible)
{	
	pChartArray->SetTraceVisibility(traceNum, isVisible);
}



DllExport void AppendData(CudaChartArray* pChartArray, int* xArray, int* yArray, int numPoints, int traceNum)
{	
	int2* newPoints = (int2*)malloc(numPoints * sizeof(int2));

	for (int i = 0; i < numPoints; i++)
	{
		newPoints[i].x = xArray[i];
		newPoints[i].y = yArray[i];
	}

	pChartArray->AppendData(newPoints, traceNum);

	free(newPoints);
}

DllExport void Redraw(CudaChartArray* pChartArray)
{
	pChartArray->Redraw();
}

DllExport void RedrawAggregate(CudaChartArray* pChartArray)
{
	pChartArray->RedrawAggregate();
}

DllExport void* GetChartImagePtr(CudaChartArray* pChartArray)
{
	return (void*)pChartArray->GetChartImagePtr();
}

DllExport void* GetRangePtr(CudaChartArray* pChartArray)
{
	range[0] = pChartArray->m_x_min;
	range[1] = pChartArray->m_x_max;
	range[2] = pChartArray->m_y_min;
	range[3] = pChartArray->m_y_max;

	return (void*)&range;
}


DllExport void* GetAggregateImagePtr(CudaChartArray* pChartArray)
{
	return (void*)pChartArray->GetAggregateImagePtr();
}


DllExport void* GetSelectionArrayPtr(CudaChartArray* pChartArray)
{
	return (void*)pChartArray->GetSelectionArrayPtr();
}


DllExport void SetWindowBackground(CudaChartArray* pChartArray, uchar4 color)
{
	pChartArray->SetWindowBackground(color);
}


DllExport void SetInitialRanges(CudaChartArray* pChartArray, int xmin, int xmax, int ymin, int ymax)
{
	pChartArray->SetInitialRanges(xmin, xmax, ymin, ymax);
}


DllExport int32_t GetRowFromY(CudaChartArray* pChartArray, int32_t y)
{
	return pChartArray->GetRowFromY(y);
}

DllExport int32_t GetColumnFromX(CudaChartArray* pChartArray, int32_t x)
{
	return pChartArray->GetColumnFromX(x);
}

DllExport void Reset(CudaChartArray* pChartArray)
{
	pChartArray->Reset();
}

#pragma endregion CudaPlot


/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

#pragma region CudaImage


DllExport uint16_t* SetFullGrayscaleImage(CudaImage* pCudaImage, uint16_t* grayImage, uint16_t imageWidth, uint16_t imageHeight)
{
	return pCudaImage->SetFullGrayscaleImage(grayImage, imageWidth, imageHeight);	
}


DllExport uint16_t* SetRoiGrayscaleImage(CudaImage* pCudaImage, uint16_t* roiImage, uint16_t imageWidth, uint16_t imageHeight, 
										uint16_t roiWidth, uint16_t roiHeight, uint16_t roiX, uint16_t roiY)
{
	return pCudaImage->SetRoiGrayscaleImage(roiImage, imageWidth, imageHeight, roiWidth, roiHeight, roiX, roiY);	
}

DllExport uint16_t* GetGrayscaleImagePtr(CudaImage* pCudaImage)
{
	return pCudaImage->mp_d_grayImage;
}

DllExport uint16_t* SetMaskImage(CudaImage* pCudaImage, uint16_t* maskImage, uint16_t maskWidth, uint16_t maskHeight, uint16_t maskRows, uint16_t maskCols)
{
	return pCudaImage->SetMaskImage(maskImage, maskWidth, maskHeight, maskRows, maskCols);
}

DllExport uint16_t* GetMaskImagePtr(CudaImage* pCudaImage)
{
	return pCudaImage->mp_d_maskImage;
}

DllExport void SetColorMap(CudaImage* pCudaImage, uint8_t* redMap, uint8_t* greenMap, uint8_t* blueMap, uint16_t maxPixelValue)
{
	pCudaImage->SetColorMap(redMap, greenMap, blueMap, maxPixelValue);	
}

DllExport uint8_t* ConvertGrayscaleToColor(CudaImage* pCudaImage, uint16_t scaleLower, uint16_t scaleUpper)
{
	return pCudaImage->ConvertGrayscaleToColor(scaleLower, scaleUpper);
}

DllExport uint8_t* GetColorImagePtr(CudaImage* pCudaImage)
{
	return pCudaImage->mp_d_colorImage;
}

DllExport void ApplyMaskToImage(CudaImage* pCudaImage)
{
	pCudaImage->ApplyMaskToImage();
}

DllExport uint8_t* PipelineFullImage(CudaImage* pCudaImage, uint16_t* grayImage, uint16_t imageWidth, uint16_t imageHeight, bool applyMask)
{
	return pCudaImage->PipelineFullImage(grayImage, imageWidth, imageHeight, applyMask);	
}

DllExport uint8_t* PipelineRoiImage(CudaImage* pCudaImage, uint16_t* roiImage, uint16_t imageWidth, uint16_t imageHeight, 
	uint16_t roiWidth, uint16_t roiHeight, uint16_t roiX, uint16_t roiY, bool applyMask)
{
	return pCudaImage->PipelineRoiImage(roiImage, imageWidth, imageHeight, roiWidth, roiHeight, roiX, roiY, applyMask);
}

DllExport void DownloadColorImage(CudaImage* pCudaImage, uint8_t* colorImageDest)
{
	cudaMemcpy(colorImageDest, pCudaImage->mp_d_colorImage, pCudaImage->m_imageW*pCudaImage->m_imageH * 4, cudaMemcpyDeviceToHost);
}

DllExport void DownloadGrayscaleImage(CudaImage* pCudaImage, uint16_t* grayImageDest)
{
	cudaMemcpy(grayImageDest, pCudaImage->mp_d_grayImage, pCudaImage->m_imageW * pCudaImage->m_imageH * sizeof(UINT16), cudaMemcpyDeviceToHost);
}


DllExport void InitImageTool(CudaImage** pp_CudaImage)
{
	*pp_CudaImage = new CudaImage();
	(*pp_CudaImage)->Init();	
}

DllExport void Shutdown_ImageTool(CudaImage* pCudaImage)
{
	pCudaImage->Shutdown();
}


DllExport void GetHistogram_512Buckets(CudaImage* pCudaImage, uint32_t* destHist, uint8_t maxValueBitWidth)
{
	pCudaImage->GetHistogram_512Buckets(destHist, maxValueBitWidth);
}

DllExport void GetHistogramImage_512Buckets(CudaImage* pCudaImage, uint8_t* histImage, uint16_t width, uint16_t height, uint32_t maxBinCount)
{
	// NOTE:  GetHistogram_512Buckets MUST BE CALLED BEFORE CALLING THIS FUNCTION!!
	pCudaImage->GetHistogramImage_512Buckets(histImage, width, height, maxBinCount);
}


DllExport void CalculateMaskApertureSums(CudaImage* pCudaImage, uint32_t* sums)
{
	pCudaImage->CalculateMaskApertureSums(sums);
}


DllExport void SetFlatFieldCorrectionArrays(CudaImage* pCudaImage, int type, float* Gc, float* Dc, int numElements)
{
	pCudaImage->SetFlatFieldCorrectionArrays(type, Gc, Dc, numElements);
}


DllExport void FlattenImage(CudaImage* pCudaImage, int type)
{
	pCudaImage->FlattenImage(type);
}


DllExport void GetImageAverage(CudaImage* pCudaImage, uint16_t* grayImage, int width, int height, int* pAverage)
{
	uint64_t sum = pCudaImage->SumImage(grayImage, width, height);

	uint64_t numElements = (uint64_t)(width*height);
	*pAverage = (int)( sum / numElements );
}


DllExport void GetGrayImageAverage(CudaImage* pCudaImage, int* pAverage)
{
	// this function assumes that a gray image has already been loaded onto the gpu using SetFullGrayscaleImage() above

	uint64_t sum = pCudaImage->SumLoadedGrayImage();

	uint64_t numElements = (uint64_t)(pCudaImage->m_imageW * pCudaImage->m_imageH);
	*pAverage = (int)(sum / numElements);
}


#pragma endregion CudaImage



#pragma region CudaUtil



DllExport void InitCudaUtil(CudaUtil** pp_CudaUtil)
{
	*pp_CudaUtil = new CudaUtil();	
}

DllExport void Shutdown_CudaUtil(CudaUtil* pCudaUtil)
{
	delete pCudaUtil;
}




//bool GetCudaDeviceCount(int &count);
DllExport int GetCudaDeviceCount(CudaUtil* pCudaUtil)
{
	int count = 0;
	if (pCudaUtil->GetCudaDeviceCount(count))
	{
		return count;
	}
	else
		return 0;
}


//bool GetComputeCapability(int &major, int &minor);
DllExport void GetCudaComputeCapability(CudaUtil* pCudaUtil, int* major, int* minor)
{
	int maj = 0, min = 0;
	if (pCudaUtil->GetComputeCapability(maj, min))
	{
		*major = maj;
		*minor = min;
	}
	else
	{
		*major = 0;
		*minor = 0;
	}
}


//bool GetDeviceName(std::string &name);
DllExport void GetCudaDeviceName(CudaUtil* pCudaUtil, char* pName, int* pLen)
{
	std::string name;
	if (pCudaUtil->GetDeviceName(name))
	{
		// copying the contents of the string to char array 
		strcpy(pName, name.c_str());
		*pLen = name.length();
	}
	else
	{
		*pLen = 0;
	}
}

//bool GetDeviceMemory(size_t &totalMem, size_t &freeMem);
DllExport void GetCudaDeviceMemory(CudaUtil* pCudaUtil, int64_t* pTotMem, int64_t* pFreeMem)
{
	size_t totmem = 0, freemem = 0;
	if (pCudaUtil->GetDeviceMemory(totmem, freemem))
	{
		*pTotMem = (int64_t)totmem;
		*pFreeMem = (int64_t)freemem;
	}
	else
	{
		*pTotMem = 0;
		*pFreeMem = 0;
	}
}


//std::string GetLastErrorMessage();
DllExport void GetCudaLastErrorMessage(CudaUtil* pCudaUtil, char* pMessage, int* pLen)
{	
	std::string msg = pCudaUtil->GetLastErrorMessage();
	
	// copying the contents of the string to char array 
	strcpy(pMessage, msg.c_str());
	*pLen = msg.length();	
}



#pragma endregion CudaUtil