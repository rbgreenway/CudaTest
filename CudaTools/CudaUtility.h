#ifndef __CUDA_UTILITY_H__
#define __CUDA_UTILITY_H__

#include <string>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

/**
* Execute a CUDA call and print out any errors
* @return the original cudaError_t result
*/
#define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)

/**
* Evaluates to true on success
*/
#define CUDA_SUCCESS(x)			(CUDA(x) == cudaSuccess)

/**
* Evaluates to true on failure
*/
#define CUDA_FAILED(x)			(CUDA(x) != cudaSuccess)

/**
* Return from the boolean function if CUDA call fails
*/
#define CUDA_VERIFY(x)			if(CUDA_FAILED(x))	return false;

/**
* LOG_CUDA string.
*/
#define LOG_CUDA "[cuda]   "

/*
* define this if you want all cuda calls to be printed
*/
//#define CUDA_TRACE



/**
* cudaCheckError
*/
inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line)
{
#if !defined(CUDA_TRACE)
	if (retval == cudaSuccess)
		return cudaSuccess;
#endif

	//int activeDevice = -1;
	//cudaGetDevice(&activeDevice);

	//Log("[cuda]   device %i  -  %s\n", activeDevice, txt);

	printf(LOG_CUDA "%s\n", txt);


	if (retval != cudaSuccess)
	{
		printf(LOG_CUDA "   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
		printf(LOG_CUDA "   %s:%i\n", file, line);
	}

	return retval;
}



struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float ElapsedMillis()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};



class CudaUtil
{
public:
	CudaUtil();
	~CudaUtil();

	//std::string GetCudaErrorMessage(CUresult cudaResult);
	//static std::string GetCudaErrorDescription(CUresult result);
	bool GetCudaDeviceCount(int &count);
	bool GetComputeCapability(int &major, int &minor);
	bool GetDeviceName(std::string &name);
	bool GetDeviceMemory(uint64_t &totalMem, uint64_t &freeMem);
	std::string GetLastErrorMessage();



private:
	CUresult m_result;
	CUdevice           m_cudaDevice;
	std::string m_errMsg;
	int  m_deviceCount;
};


#endif
