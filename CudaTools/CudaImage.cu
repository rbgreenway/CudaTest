#include "CudaImage.h"

#define LOG_CUDA


#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Constructor
CudaImage::CudaImage()
{
	
}

CudaImage::~CudaImage()
{

}





/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Cuda Kernels

__global__ void Compute_Histogram_512_Cuda(uint32_t* hist, const uint16_t* data, uint16_t width, uint16_t height, uint8_t maxValueBitWidth)
{
	// NOTE: # of bins of histogram must match block size (number of threads in block), and in this case must be 512.
	//		 i.e. the number of threads per block must be the same as the number of bins.

	// maxValueBitWidth = the number of bits needed to represent the max value in the data array.  For example, if the data
	//					  array is built from a 10-bit A-to-D converter, then maxValueBitWidth = 10 since no value will be greather 
	//					  than 2^10.  The minimum value for maxValueBitWidth is driven by the number of bins.  For 256 bins (2^8), 
	//					  the min value is 8.  If bins were 1024 (i.e. 2^10), then the min value for maxValueBitWidth would be 10.

	if (maxValueBitWidth < 8) maxValueBitWidth = 8; // make sure we aren't below the min as described above

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int nThread = threadIdx.y * blockDim.x + threadIdx.x; // index of thread within block
	int nPixel = y * width + x; // index of pixel within image

	if (x >= width) return;
	if (y >= height) return;

	// if image pixel value == 0, don't add it to the histogram.  Pixel that are 0 are pixels that are outside of the mask 
	// and thus should not be part of the histogram
	if (data[nPixel] == 0) return;

	//Create shared buffer size of threads per block and clear it 
	//Size of array equals numBins 
	__shared__ uint32_t tmpHist[512];
	tmpHist[nThread] = 0;
	__syncthreads();


	//based on the value of this pixel, find the correct bin of the local histogram to increment, and then increment it
	uint8_t shift = maxValueBitWidth - 9;
	int binNumber = data[nPixel] >> shift;

	if (binNumber>511)
	{
		binNumber = 511;
	}

	//float f1 = ((float)(data[nPixel]))/1023.0 * 255;
	//uint8_t binNumber = (uint8_t)f1;


	atomicAdd(&(tmpHist[binNumber]), 1);
	__syncthreads();  // wait for all threads in this block to finish so that the local histogram is finished

					  // Update global memory (global histogram)	
	atomicAdd(&(hist[nThread]), tmpHist[nThread]);

}

__global__ void compute_histogram_256_Cuda(uint32_t* hist, const uint16_t* data, uint16_t width, uint16_t height, uint8_t maxValueBitWidth)
{
	// NOTE: # of bins of histogram must match block size (number of threads in block), and in this case must be 256.
	//		 i.e. the number of threads per block must be the same as the number of bins.

	// maxValueBitWidth = the number of bits needed to represent the max value in the data array.  For example, if the data
	//					  array is built from a 10-bit A-to-D converter, then maxValueBitWidth = 10 since no value will be greather 
	//					  than 2^10.  The minimum value for maxValueBitWidth is driven by the number of bins.  For 256 bins (2^8), 
	//					  the min value is 8.  If bins were 1024 (i.e. 2^10), then the min value for maxValueBitWidth would be 10.

	if (maxValueBitWidth < 8) maxValueBitWidth = 8; // make sure we aren't below the min as described above

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int nThread = threadIdx.y * blockDim.x + threadIdx.x; // index of thread within block
	int nPixel = y * width + x; // index of pixel within image

	if (x >= width) return;
	if (y >= height) return;

	//Create shared buffer size of threads per block and clear it 
	//Size of array equals numBins 
	__shared__ uint32_t tmpHist[256];
	tmpHist[nThread] = 0;
	__syncthreads();


	//based on the value of this pixel, find the correct bin of the local histogram to increment, and then increment it
	uint8_t shift = maxValueBitWidth - 8;
	int binNumber = data[nPixel] >> shift;

	if (binNumber>255)
	{
		binNumber = 255;
	}

	//float f1 = ((float)(data[nPixel]))/1023.0 * 255;
	//uint8_t binNumber = (uint8_t)f1;


	atomicAdd(&(tmpHist[binNumber]), 1);
	__syncthreads();  // wait for all threads in this block to finish so that the local histogram is finished

					  // Update global memory (global histogram)	
	atomicAdd(&(hist[nThread]), tmpHist[nThread]);

}

__global__ void MaskImage_Cuda(uint16_t* image, uint16_t* mask, uint16_t width, uint16_t height)
{
	// this function zeroes out all pixels in image that are not in the mask

	// image - a greyscale image with each pixel being a uint16_t
	// mask - a image where pixels with value>0 will be passed through, and pixels with value==0 will be masked out (set to zero).
	//		  The mask is created where pixels with a value of 1, belong in mask aperture 1.  Pixels with value of 2, belong in 
	//		  mask aperture 2...and so on.  
	// width,height - dimensions of image in pixels

	// calc x,y position of pixel to operate on
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; // column of pixel inside panel
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; // row of pixel inside panel

														// make sure we don't try to operate outside the image
	if (x >= width) return;
	if (y >= height) return;

	// calculate pixel position in array
	uint32_t n = (y * width) + x;

	// apply mask to image
	if (mask[n] == 0)
	{
		// this pixel is not within a mask aperture, so zero it out
		image[n] = 0;
	}
}

__global__ void FlattenImage_Cuda(uint16_t* image, float* Gc, float* Dc, uint16_t width, uint16_t height)
{
	// this function flattens the image using

	// image - a greyscale image with each pixel being a uint16_t	
	// width,height - dimensions of image in pixels
	// Equation:
	//              flattenedImage[n] = (inputImage[n] - Dc[n]) * Gc[n]
	//	
	//
	//  C = corrected image  (Cij = the pixel at column i and row j)	
	//  D = dark image (this is an image taken with no lighting.  it bascially gives the dark current noise)
	//  G = gain
	//  Dc = D corrected to binning size
	//  Gc = G corrected to binning size


	// calc x,y position of pixel to operate on
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; // column of pixel inside panel
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; // row of pixel inside panel

														// make sure we don't try to operate outside the image
	if (x >= width) return;
	if (y >= height) return;

	// calculate pixel position in array
	uint32_t n = (y * width) + x;

	// adjust pixel to flatten image
	float fval = ((float)image[n] - Dc[n]) * Gc[n];
	if (fval < 0.0f) fval = 0.0f;
	if (fval > 65535.0f) fval = 65535.0f;

	image[n] = (uint16_t)fval;
}


__global__ void ConvertGrayscaleToColor_Cuda(uint8_t* color, uint16_t* gray, uint8_t* redMap, uint8_t* greenMap, uint8_t* blueMap,
	uint16_t width, uint16_t height, uint16_t maxGrayValue, uint16_t scaleLower, uint16_t scaleUpper)
{
	// this function converts a grayscale image to a color image using the provided color map

	// color - destination color image (format is ARGB)
	// gray -  source grayscale image
	// redMap, greenMap, blueMap - arrays (maps) that provide color components for each possible grayscale value. For example,
	//							   if a pixel in the gray image has a value = 100, then the corresponding pixel in the color image
	//							   would have its RGB component values set to redMap[100], greenMap[100], and blueMap[100], respectively.
	// width, height - image dimensions
	// maxGrayValue - the maximum possible grayscale value, i.e. length of color map (length of redMap, greenMap, and blueMap)

	// scaleLower, scaleUpper - these values are used to scale the grayscale value of a pixel before it is converted to color.
	//
	//                         scaleUpper
	//						   ________________
	//	maxGrayValue|         /
	//				|        /
	//				|       /
	//				|      /
	//			0	|_____/____________________ 
	//                   scaleLower
	//
	//  Here's the math:
	//		if (pixelValue < scaleLower) set pixelValue = 0
	//      else if (pixelValue < scaleUpper) set pixelValue = maxGrayValue
	//      else 

	// calc x,y position of pixel to operate on
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; // column of pixel inside panel
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; // row of pixel inside panel

														// make sure we don't try to operate outside the image
	if (x >= width) return;
	if (y >= height) return;

	// calculate pixel position in gray array
	uint32_t nG = (y * width) + x;

	// calculate pixel position in color array
	uint32_t nC = (y * width * 4) + (x * 4);

	// make sure grayscale value is not outside of color maps
	if (gray[nG] > maxGrayValue) gray[nG] = maxGrayValue;

	// scale the value
	uint16_t val = gray[nG];
	if (val < scaleLower) val = 0;
	else if (val >= scaleUpper) val = maxGrayValue;
	else
	{
		float fval = (float)maxGrayValue / (float)(scaleUpper - scaleLower) * (float)(val - scaleLower);
		val = (uint16_t)fval;
	}

	// set pixel component values for color image
	color[nC + 0] = blueMap[val];	// blue
	color[nC + 1] = greenMap[val];	// green
	color[nC + 2] = redMap[val];	// red
	color[nC + 3] = 255;			// alpha

}

__global__ void CopyCudaArrayToD3D9Memory_Cuda(uint8_t *dest, uint8_t *source, uint16_t pitch, uint16_t width, uint16_t height)
{
	// calc x,y position of pixel to operate on
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; // column of pixel inside panel
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; // row of pixel inside panel

														// make sure we don't try to operate outside the image
	if (x >= width) return;
	if (y >= height) return;

	// calc position of pixel in cuda array (remember that pitch may not equal width)
	//uint32_t nD = ((height - 1 - y)*pitch) + (x * 4);
	uint32_t nD = (y*pitch) + (x * 4);
	uint32_t nS = (y*width * 4) + (x * 4);

	// copy data
	dest[nD] = source[nS];
	dest[nD + 1] = source[nS + 1];
	dest[nD + 2] = source[nS + 2];
	dest[nD + 3] = source[nS + 3];
}

__global__ void BuildHistogramImage_Cuda(uint8_t* histImage, uint32_t* hist, uint16_t numBins, uint16_t width, uint16_t height, uint32_t maxBinCount)
{
	// this function builds the image for a histogram given by the variable hist.  
	//
	// histImage - the output histogram image.  This is a color image (ARGB, 8 bits per component)
	// hist - is an array which contains the data for the histogram
	// numBins - is the number of bins in the histogram
	// width, height - dimensions of the histImage in pixels
	// maxBinCount - the maximum value that can appear in each bin of the histogram

	// calc x,y position of pixel to operate on
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; // column of pixel inside panel
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; // row of pixel inside panel

														// make sure we don't try to operate outside the image
	if (x >= width) return;
	if (y >= height) return;

	// calculate the array index into the histogram image
	uint32_t n = (y * width * 4) + (x * 4);  // ARGB image

											 // calculate the width of each bin in pixels
	uint16_t binWidth = width / numBins;

	// calculate the bin that this pixel belongs in
	uint16_t binNumber = x / binWidth;
	if (binNumber>numBins) binNumber = numBins;

	// calculate height of the bar for his bin
	uint32_t value = hist[binNumber];  // get the height of the bar for this bin
	uint32_t barHeight = (uint32_t)((float)value * (float)height / (float)maxBinCount);  // calculate the bar height in pixels
	if (barHeight > height) barHeight = height; // make sure the bar height in pixels is not greater than the histogram image height

												// determine if this pixel is in the bar or above it (i.e. determine color of pixel)
	if (y < (height - barHeight)) // pixel is above bar (thus pixel is background color...likely white)
	{
		histImage[n + 0] = 220;	// blue
		histImage[n + 1] = 220;	// green
		histImage[n + 2] = 220;	// red
		histImage[n + 3] = 255;	// alpha
	}
	else  // pixel is part of bar, so make it the color of the bar (likely black)
	{
		histImage[n + 0] = 0;	// blue
		histImage[n + 1] = 0;	// green
		histImage[n + 2] = 0;	// red
		histImage[n + 3] = 255;	// alpha
	}
}

__global__ void CalcApertureSums_Cuda(uint32_t* sumArray, uint16_t* image, uint16_t* mask, uint16_t width, uint16_t height)
{
	// This function calculate the sum of pixels for each aperture of a mask.  It expects that the mask is formated as follows:
	//		mask pixels with a value of 0 belong to no apertures, thus they will not be part of any sum
	//      mask pixels with a value of 1 belong in aperture 1, which is added to the value in sumArray[0]
	//      mask pixels with a value of 2 belong in aperture 2, which is added to the value in sumArray[1]
	//		and so on...

	// sumArray - output array of the sum of pixel values for each aperature.  For example, for a mask with 24x16 (384) apertures, there
	//			  will be 384 values in sumArray
	// image - input grayscale image from which sums are calculated
	// mask  - input mask that is formatted as described in the description above for this function
	// width, height - dimensions of the image and mask in pixels

	// calc x,y position of pixel to operate on
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; // column of pixel inside panel
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; // row of pixel inside panel

														// make sure we don't try to operate outside the image
	if (x >= width) return;
	if (y >= height) return;

	// calculate pixel position in image and mask
	uint32_t n = (y * width) + x;

	// get aperture number from mask
	if (mask[n] > 0) // is this pixel inside of any of the apertures of the mask?
	{ // yes
		atomicAdd(&sumArray[mask[n] - 1], image[n]);
	}

	__syncthreads();  // wait for all threads in this block to finish so that the local histogram is finished

}

__global__ void FlatField_Cuda(uint16_t* image, uint16_t* dark, uint16_t* gain, uint16_t width, uint16_t height)
{
	// this function flat field corrects the given grayscale image. It uses the following function:
	//
	//		C[i,j] = ((R[i,j] - D[i,j]) * m) / (F[i,j] - D[i,j]) = (R[i,j] - D[i,j]) * G[i,j]
	//
	//			where G[i,j] = m / (F[i,j] - D[i,j])
	//
	//				  m = average of F-D
	//
	//		i,j = row,column of pixel in image
	//		C = corrected image
	//		R = raw image
	//		F = flat field reference image (evenly illuminated image, meant to show unevenness of illumination)
	//		D = dark field reference image (image taken with no illumination, meant to show distribution of background)
	//		G = gain

	//	parameters passed into function:
	//	image - grayscale image to be corrected.  This is both the input and output image (the input image is over written)
	//  dark  - this is the dark field image (must be same dimensions as image), probably stored in database
	//  gain  - this is the gain array (must be same dimensions as image), that is calculated elsewhere
	//  width, height - dimensions of image (and dark) in pixels

	// calc x,y position of pixel to operate on
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; // column of pixel inside panel
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; // row of pixel inside panel

														// make sure we don't try to operate outside the image
	if (x >= width) return;
	if (y >= height) return;

	// calculate pixel position in image and dark arrays
	uint32_t n = (y * width) + x;

	image[n] = (image[n] - dark[n]) * gain[n];
}

__global__ void CopyRoiToFullImage_Cuda(uint16_t* full, uint16_t* roi, uint16_t fullW, uint16_t fullH,
	uint16_t  roiX, uint16_t roiY, uint16_t roiW, uint16_t roiH)
{
	// This function is used to copy a ROI image from the camera into a memory space that holds a full frame.
	// It is used when the camera is set up to capture only a part of the CCD (an Region of Interest - ROI), and 
	// since all of the algorithms, kernels, display routines, etc. are set up to handle full frames, this
	// function simply copies the ROI into a full frame.  Pixels outside the ROI are set to zero.

	// calc x,y position of pixel to operate on in the full frame
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; // column of pixel inside full frame image
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; // row of pixel inside full frame image

														// make sure we don't try to operate outside the full image
	if (x >= fullW) return;
	if (y >= fullH) return;

	// calculate pixel position in arrays
	uint32_t fullN = (y * fullW) + x;  // index into full frame

									   // calculate x,y position in ROI
	int32_t xr = x - roiX;
	int32_t yr = y - roiY;

	// are we inside ROI?

	if (x >= roiX && x < (roiX + roiW) && y >= roiY && y < (roiY + roiH))
	{
		uint32_t roiN = (yr * roiW) + xr; // index into roi frame

										  // inside ROI
		full[fullN] = roi[roiN];
	}
	else
	{
		// outside ROI
		full[fullN] = 0;
	}
}




template <unsigned int blockSize> __device__ void warpReduce(volatile uint64_t *sdata, unsigned int tid)
{
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}


template <unsigned int blockSize> __global__ void reduce6(uint16_t *g_idata, uint64_t *g_odata, unsigned int n)
{
	extern __shared__ uint64_t sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;

	while (i < n)
	{
		sdata[tid] += g_idata[i] + g_idata[i + blockSize];  
		i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] += sdata[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (tid < 64)
		{
			sdata[tid] += sdata[tid + 64];
		}
		__syncthreads();
	}
	if (tid < 32) warpReduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
	__device__ inline operator double *()
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}

	__device__ inline operator const double *() const
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}
};


/*
This version adds multiple elements per thread sequentially.  This reduces the overall
cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
(Brent's Theorem optimization)

Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T1, class T2, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T1 *g_idata, T2 *g_odata, unsigned int n)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T2 *sdata = SharedMemory<T2>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T2 mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += g_idata[i + blockSize];

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	cg::sync(cta);


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	cg::sync(cta);

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 64];
	}

	cg::sync(cta);

#if (__CUDA_ARCH__ >= 300 )
	if (tid < 32)
	{
		cg::coalesced_group active = cg::coalesced_threads();

		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			mySum += active.shfl_down(mySum, offset);
		}
	}
#else
	// fully unroll reduction within a single warp
	if ((blockSize >= 64) && (tid < 32))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 32];
	}

	cg::sync(cta);

	if ((blockSize >= 32) && (tid < 16))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 16];
	}

	cg::sync(cta);

	if ((blockSize >= 16) && (tid <  8))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 8];
	}

	cg::sync(cta);

	if ((blockSize >= 8) && (tid <  4))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 4];
	}

	cg::sync(cta);

	if ((blockSize >= 4) && (tid <  2))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 2];
	}

	cg::sync(cta);

	if ((blockSize >= 2) && (tid <  1))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 1];
	}

	cg::sync(cta);
#endif

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = mySum;
}



bool isPow2(unsigned int x)
{
	return ((x&(x - 1)) == 0);
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T1, class T2>  // T1 is input type, T2 is output type
void
reduce(int size, int threads, int blocks, T1 *d_idata, T2 *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T2) : threads * sizeof(T2);


		if (isPow2(size))
		{
			switch (threads)
			{
			case 512:
				reduce6<T1, T2, 512, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case 256:
				reduce6<T1, T2, 256, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case 128:
				reduce6<T1, T2, 128, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case 64:
				reduce6<T1, T2, 64, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case 32:
				reduce6<T1, T2, 32, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case 16:
				reduce6<T1, T2, 16, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case  8:
				reduce6<T1, T2, 8, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case  4:
				reduce6<T1, T2, 4, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case  2:
				reduce6<T1, T2, 2, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case  1:
				reduce6<T1, T2, 1, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;
			}
		}
		else
		{
			switch (threads)
			{
			case 512:
				reduce6<T1, T2, 512, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case 256:
				reduce6<T1, T2, 256, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case 128:
				reduce6<T1, T2, 128, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case 64:
				reduce6<T1, T2, 64, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case 32:
				reduce6<T1, T2, 32, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case 16:
				reduce6<T1, T2, 16, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case  8:
				reduce6<T1, T2, 8, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case  4:
				reduce6<T1, T2, 4, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case  2:
				reduce6<T1, T2, 2, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;

			case  1:
				reduce6<T1, T2, 1, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
				break;
			}
		}

}



uint64_t CudaImage::SumImage(uint16_t* grayImage, uint16_t width, uint16_t height)
{
	unsigned int elementCount = ((unsigned int)width)*((unsigned int)height);
	unsigned int blockSize = 256;  // number of threads
	unsigned int numBlocks = (elementCount + blockSize - 1) / blockSize;

	// copy gray (input) image to gpu
	uint16_t* d_grayImage;
	cudaError_t res = cudaMalloc(&d_grayImage, elementCount * sizeof(uint16_t));
	res = cudaMemcpy(d_grayImage, grayImage, elementCount * sizeof(uint16_t), cudaMemcpyHostToDevice);

	// allocate for output data (of type uint64_t), and initialize to zero
	uint64_t* d_output;
	res = cudaMalloc(&d_output, numBlocks * sizeof(uint64_t));
	cudaMemset(d_output, 0, numBlocks * sizeof(uint64_t));

	reduce<uint16_t, uint64_t> (elementCount, blockSize, numBlocks, d_grayImage, d_output);

	uint64_t *p_vals = (uint64_t*)malloc(numBlocks*sizeof(uint64_t));

	res = cudaMemcpy(p_vals, d_output, numBlocks * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	uint64_t sum = 0;

	for (int i = 0; i < numBlocks; i++)
		sum += p_vals[i];


	cudaFree(d_grayImage);
	cudaFree(d_output);
	free(p_vals);

	return sum;
}


uint64_t CudaImage::SumLoadedGrayImage()
{
	if (mp_d_grayImage == 0) return 0;

	unsigned int elementCount = ((unsigned int)m_imageW)*((unsigned int)m_imageH);
	unsigned int blockSize = 512;  // number of threads
	unsigned int numBlocks = (elementCount + blockSize - 1) / blockSize;

	// allocate for output data (of type uint64_t), and initialize to zero
	uint64_t* d_output;
	cudaError_t res = cudaMalloc(&d_output, numBlocks * sizeof(uint64_t));
	cudaMemset(d_output, 0, numBlocks * sizeof(uint64_t));

	reduce<uint16_t, uint64_t>(elementCount, blockSize, numBlocks, mp_d_grayImage, d_output);

	uint64_t *p_vals = (uint64_t*)malloc(numBlocks * sizeof(uint64_t));

	res = cudaMemcpy(p_vals, d_output, numBlocks * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	uint64_t sum = 0;

	for (int i = 0; i < numBlocks; i++)
		sum += p_vals[i];

	cudaFree(d_output);
	free(p_vals);

	return sum;
}


void CudaImage::Test()
{
	uint16_t w = 1024;
	uint16_t h = 1024;

	uint16_t* p_data = (uint16_t*)malloc(w*h * sizeof(uint16_t));

	for (int r = 0; r < h; r++)
	{
		for (int c = 0; c < w; c++)
		{
			int index = r*w + c;

			p_data[index] = 1;
		}
	}


	GpuTimer t1;

	t1.Start();
	uint64_t sum = SumImage(p_data, w, h);
	t1.Stop();
	float t = t1.ElapsedMillis();

	free(p_data);

	float avg = (float)sum / (float)(w*h);

	sum += 0;


}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel Calling Functions

void CudaImage::ConvertGrayscaleToColor(uint8_t* color, uint16_t* gray, uint8_t* redMap, uint8_t* greenMap, uint8_t* blueMap,
	uint16_t width, uint16_t height, uint16_t maxGrayValue, uint16_t scaleLower, uint16_t scaleUpper)
{
	dim3 block, grid;
	block.x = 32; block.y = 8; block.z = 1;
	grid.x = width / block.x;
	grid.y = height / block.y;
	grid.z = 1;

	ConvertGrayscaleToColor_Cuda << <grid, block >> >(color, gray, redMap, greenMap, blueMap, width, height, maxGrayValue, scaleLower, scaleUpper);

}

void CudaImage::CopyRoiToFullImage(uint16_t* full, uint16_t* roi, uint16_t fullW, uint16_t fullH,
	uint16_t  roiX, uint16_t roiY, uint16_t roiW, uint16_t roiH)
{
	dim3 block, grid;
	block.x = 32; block.y = 8; block.z = 1;
	grid.x = fullW / block.x;
	grid.y = fullH / block.y;
	grid.z = 1;
	CopyRoiToFullImage_Cuda << <grid, block >> >(full, roi, fullW, fullH, roiX, roiY, roiW, roiH);
}

void CudaImage::MaskImage(uint16_t* image, uint16_t* mask, uint16_t width, uint16_t height)
{
	dim3 block, grid;
	block.x = 32; block.y = 8; block.z = 1;
	grid.x = width / block.x;
	grid.y = height / block.y;
	grid.z = 1;
	MaskImage_Cuda << <grid, block >> >(image, mask, width, height);
}

void CudaImage::FlattenImage(uint16_t* image, float* Gc, float* Dc, uint16_t width, uint16_t height)
{
	dim3 block, grid;
	block.x = 32; block.y = 8; block.z = 1;
	grid.x = width / block.x;
	grid.y = height / block.y;
	grid.z = 1;
	FlattenImage_Cuda << <grid, block >> >(image, Gc, Dc, width, height);
}

void CudaImage::CopyCudaArrayToD3D9Memory(uint8_t* pDest, uint8_t* pSource, uint16_t pitch, uint16_t width, uint16_t height)
{
	cudaError_t res = cudaDeviceSynchronize();

	dim3 threadsPerBlock(32, 32);  // 32x16 = 512 threads per block	
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	CopyCudaArrayToD3D9Memory_Cuda << <numBlocks, threadsPerBlock >> >(pDest, pSource, pitch, width, height);
}

void CudaImage::ComputeHistogram_512(uint32_t* hist, const uint16_t* data, uint16_t width, uint16_t height, uint8_t maxValueBitWidth)
{
	dim3 block, grid;
	block.x = 32; block.y = 16; block.z = 1; // block size must be 512 = 32 * 16
	grid.x = width / block.x;
	grid.y = height / block.y;
	grid.z = 1;

	Compute_Histogram_512_Cuda << <grid, block >> >(hist, data, width, height, maxValueBitWidth);
}

void CudaImage::BuildHistogramImage_512(uint8_t* histImage, uint32_t* hist, uint16_t numBins, uint16_t width, uint16_t height, uint32_t maxBinCount)
{
	dim3 block, grid;
	block.x = 32; block.y = 16; block.z = 1;
	grid.x = width / block.x;
	grid.y = height / block.y;
	grid.z = 1;

	BuildHistogramImage_Cuda << <grid, block >> >(histImage, hist, numBins, width, height, maxBinCount);
}

void CudaImage::CalcApertureSums(uint32_t* sumArray, uint16_t* image, uint16_t* mask, uint16_t width, uint16_t height)
{
	dim3 block, grid;
	block.x = 32; block.y = 16; block.z = 1;
	grid.x = width / block.x;
	grid.y = height / block.y;
	grid.z = 1;

	CalcApertureSums_Cuda << <grid, block >> >(sumArray, image, mask, width, height);
}





uint16_t* CudaImage::SetFullGrayscaleImage(uint16_t* grayImage, uint16_t imageWidth, uint16_t imageHeight)
{
	if (imageWidth != m_imageW || imageHeight != m_imageH)
	{
		if (mp_d_grayImage != 0) cudaFree(mp_d_grayImage);
		if (mp_d_colorImage != 0) cudaFree(mp_d_colorImage);

		m_imageW = imageWidth;
		m_imageH = imageHeight;
		cudaError res = cudaMalloc(&mp_d_grayImage, m_imageW*m_imageH * sizeof(uint16_t));
		res = cudaMalloc(&mp_d_colorImage, m_imageW*m_imageH * 4);
	}

	cudaError_t err = cudaMemcpy(mp_d_grayImage, grayImage, m_imageW*m_imageH * sizeof(uint16_t), cudaMemcpyHostToDevice);

	return mp_d_grayImage;
}



uint16_t* CudaImage::SetRoiGrayscaleImage(uint16_t* roiImage, uint16_t imageWidth, uint16_t imageHeight, uint16_t roiWidth, 
							   uint16_t roiHeight, uint16_t roiX, uint16_t roiY)
{
	if (imageWidth != m_imageW || imageHeight != m_imageH)
	{
		if (mp_d_grayImage != 0) cudaFree(mp_d_grayImage);
		if (mp_d_colorImage != 0) cudaFree(mp_d_colorImage);

		m_imageW = imageWidth;
		m_imageH = imageHeight;
		cudaError res = cudaMalloc(&mp_d_grayImage, m_imageW*m_imageH * sizeof(uint16_t));
		res = cudaMalloc(&mp_d_colorImage, m_imageW*m_imageH * 4);
	}

	if (roiWidth != m_roiW || roiHeight != m_roiH || roiX != m_roiX || roiY != m_roiY)
	{
		if (mp_d_roiImage != 0) cudaFree(mp_d_roiImage);

		m_roiW = roiWidth;
		m_roiH = roiHeight;
		m_roiX = roiX;
		m_roiY = roiY;
		cudaMalloc(&mp_d_roiImage, m_roiW*m_roiH * sizeof(uint16_t));
	}

	cudaError_t errNo = cudaMemcpy(mp_d_roiImage, roiImage, m_roiW*m_roiH * sizeof(uint16_t), cudaMemcpyHostToDevice);

	CopyRoiToFullImage(mp_d_grayImage, mp_d_roiImage, m_imageW, m_imageH, m_roiX, m_roiY, m_roiW, m_roiH);

	return mp_d_grayImage;
}



uint16_t* CudaImage::SetMaskImage(uint16_t* maskImage, uint16_t maskWidth, uint16_t maskHeight, uint16_t maskRows, uint16_t maskCols)
{
	if (m_maskW != maskWidth || m_maskH != maskHeight)
	{
		if (mp_d_maskImage != 0) cudaFree(mp_d_maskImage);

		m_maskW = maskWidth;
		m_maskH = maskHeight;
		m_maskRows = maskRows;
		m_maskCols = maskCols;
		cudaMalloc(&mp_d_maskImage, m_maskW*m_maskH * sizeof(uint16_t));
	}


	// copy mask image to GPU
	cudaMemcpy(mp_d_maskImage, maskImage, m_maskW*m_maskH * sizeof(uint16_t), cudaMemcpyHostToDevice);

	m_maskSet = true;

	return mp_d_maskImage;
}



void CudaImage::SetColorMap(uint8_t* redMap, uint8_t* greenMap, uint8_t* blueMap, uint16_t maxPixelValue)
{
	cudaError_t res;
	if (mp_d_redMap != 0) cudaFree(mp_d_redMap);
	if (mp_d_greenMap != 0) cudaFree(mp_d_greenMap);
	if (mp_d_blueMap != 0) cudaFree(mp_d_blueMap);

	res = cudaMalloc(&mp_d_redMap, maxPixelValue + 1);
	res = cudaMalloc(&mp_d_greenMap, maxPixelValue + 1);
	res = cudaMalloc(&mp_d_blueMap, maxPixelValue + 1);

	res = cudaMemcpy(mp_d_redMap, redMap, maxPixelValue + 1, cudaMemcpyHostToDevice);
	res = cudaMemcpy(mp_d_greenMap, greenMap, maxPixelValue + 1, cudaMemcpyHostToDevice);
	res = cudaMemcpy(mp_d_blueMap, blueMap, maxPixelValue + 1, cudaMemcpyHostToDevice);

	m_maxPixelValue = maxPixelValue;

	m_colorMapSet = true;
}


uint8_t* CudaImage::ConvertGrayscaleToColor(uint16_t scaleLower, uint16_t scaleUpper)
{
	if (m_colorMapSet && mp_d_grayImage != 0)
	{
		if (mp_d_colorImage == 0) cudaMalloc(&mp_d_colorImage, m_imageW*m_imageH * 4);

		ConvertGrayscaleToColor(mp_d_colorImage, mp_d_grayImage, mp_d_redMap, mp_d_greenMap, mp_d_blueMap, m_imageW, m_imageH, m_maxPixelValue, scaleLower, scaleUpper);
	}

	return mp_d_colorImage;
}


void CudaImage::ApplyMaskToImage()
{
	if (m_maskSet)
	{
		MaskImage(mp_d_grayImage, mp_d_maskImage, m_imageW, m_imageH);
	}
}


uint8_t* CudaImage::PipelineFullImage(uint16_t* grayImage, uint16_t imageWidth, uint16_t imageHeight, bool applyMask)
{
	SetFullGrayscaleImage(grayImage, imageWidth, imageHeight);
	if (applyMask) ApplyMaskToImage();
	ConvertGrayscaleToColor(0, m_maxPixelValue);

	return mp_d_colorImage;
}


uint8_t* CudaImage::PipelineRoiImage(uint16_t* roiImage, uint16_t imageWidth, uint16_t imageHeight, 
	uint16_t roiWidth, uint16_t roiHeight, uint16_t roiX, uint16_t roiY, bool applyMask)
{
	SetRoiGrayscaleImage(roiImage, imageWidth, imageHeight, roiWidth, roiHeight, roiX, roiY);
	if (applyMask) ApplyMaskToImage();
	ConvertGrayscaleToColor(0, m_maxPixelValue);

	return mp_d_colorImage;
}


void CudaImage::Init()
{
	mp_d_grayImage = 0;
	mp_d_colorImage = 0;
	mp_d_maskImage = 0;
	mp_d_roiImage = 0;
	mp_d_redMap = 0;
	mp_d_greenMap = 0;
	mp_d_blueMap = 0;
	m_colorMapSet = false;
	m_maskSet = false;
	m_imageW = 0;
	m_imageH = 0;
	m_roiW = 0;
	m_roiH = 0;
	m_roiX = 0;
	m_roiY = 0;
	m_maskW = 0;
	m_maskH = 0;
	m_maskRows = 0;
	m_maskCols = 0;
	m_maxPixelValue = 65535;
	mp_d_histogram = 0;
	mp_d_colorHistogramImage = 0;

	mp_d_FFC_Fluor_Gc = 0;
	mp_d_FFC_Fluor_Dc = 0;
	mp_d_FFC_Lumi_Gc = 0;
	mp_d_FFC_Lumi_Dc = 0;
	m_h_FFC_numElements = 0;


	// not sure why I have to do this, bu
	cudaMalloc(&mp_d_grayImage, 10);
	cudaMalloc(&mp_d_colorImage, 10);
}


void CudaImage::Shutdown()
{
	if (this != 0)
	{
		if (mp_d_grayImage != 0) {
			cudaError_t err = cudaFree(mp_d_grayImage);
			mp_d_grayImage = 0;
		}
		if (mp_d_colorImage != 0) {
			cudaFree(mp_d_colorImage);
			mp_d_colorImage = 0;
		}
		if (mp_d_maskImage != 0) {
			cudaFree(mp_d_maskImage);
			mp_d_maskImage = 0;
		}
		if (mp_d_roiImage != 0) {
			cudaFree(mp_d_roiImage);
			mp_d_roiImage = 0;
		}
		if (mp_d_redMap != 0) {
			cudaFree(mp_d_redMap);
			mp_d_redMap = 0;
		}
		if (mp_d_greenMap != 0) {
			cudaFree(mp_d_greenMap);
			mp_d_greenMap = 0;
		}
		if (mp_d_blueMap != 0) {
			cudaFree(mp_d_blueMap);
			mp_d_blueMap = 0;
		}
		if (mp_d_histogram != 0) {
			cudaFree(mp_d_histogram);
			mp_d_histogram = 0;
		}
		if (mp_d_colorHistogramImage != 0) {
			cudaFree(mp_d_colorHistogramImage);
			mp_d_colorHistogramImage = 0;
		}
		if (mp_d_maskApertureSums != 0) {
			cudaFree(mp_d_maskApertureSums);
			mp_d_maskApertureSums = 0;
		}
		if (mp_d_FFC_Fluor_Gc != 0) {
			cudaFree(mp_d_FFC_Fluor_Gc);
			mp_d_FFC_Fluor_Gc = 0;
		}
		if (mp_d_FFC_Fluor_Dc != 0) {
			cudaFree(mp_d_FFC_Fluor_Dc);
			mp_d_FFC_Fluor_Dc = 0;
		}
		if (mp_d_FFC_Lumi_Gc != 0) {
			cudaFree(mp_d_FFC_Lumi_Gc);
			mp_d_FFC_Lumi_Gc = 0;
		}
		if (mp_d_FFC_Lumi_Dc != 0) {
			cudaFree(mp_d_FFC_Lumi_Dc);
			mp_d_FFC_Lumi_Dc = 0;
		}
	}
}


void CudaImage::GetHistogram_512Buckets(uint32_t* destHist, uint8_t maxValueBitWidth)
{
	if (mp_d_histogram == 0)
	{
		cudaMalloc(&mp_d_histogram, 512 * sizeof(uint32_t));
	}

	cudaMemset(mp_d_histogram, 0, 512 * sizeof(uint32_t));

	ComputeHistogram_512(mp_d_histogram, mp_d_grayImage, m_imageW, m_imageH, maxValueBitWidth);

	//cudaMemset(mp_d_histogram, 0, sizeof(uint32_t));  // zero the first bin, since that is the pixels that were masked out

	cudaMemcpy(destHist, mp_d_histogram, 512 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	m_max_histogramBinValue = 0;

	for (int i = 1; i < 512; i++)
	{
		if (destHist[i] > m_max_histogramBinValue) m_max_histogramBinValue = destHist[i];
	}

}


void CudaImage::GetHistogramImage_512Buckets(uint8_t* histImage, uint16_t width, uint16_t height, uint32_t maxBinCount)
{
	// NOTE:  GetHistogram_512Buckets MUST BE CALLED BEFORE CALLING THIS FUNCTION!!

	if (mp_d_colorHistogramImage == 0)
	{
		cudaMalloc(&mp_d_colorHistogramImage, width*height * 4);
	}

	if (maxBinCount == 0) maxBinCount = m_max_histogramBinValue;

	BuildHistogramImage_512(mp_d_colorHistogramImage, mp_d_histogram, 512, width, height, maxBinCount);

	cudaMemcpy(histImage, mp_d_colorHistogramImage, width * height * 4, cudaMemcpyDeviceToHost);
}


void CudaImage::CalculateMaskApertureSums(uint32_t* sums)
{
	if (mp_d_maskApertureSums != 0)	cudaFree(mp_d_maskApertureSums);
	uint32_t numApertures = m_maskRows * m_maskCols;
	cudaMalloc(&mp_d_maskApertureSums, numApertures * sizeof(uint32_t));
	cudaMemset(mp_d_maskApertureSums, 0, numApertures * sizeof(uint32_t));

	CalcApertureSums(mp_d_maskApertureSums, mp_d_grayImage, mp_d_maskImage, m_imageW, m_imageH);

	cudaMemcpy(sums, mp_d_maskApertureSums, numApertures * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}



void CudaImage::SetFlatFieldCorrectionArrays(int type, float* Gc, float* Dc, int numElements)
{
	// 1 = Fluor
	// 2 = Lumi

	if (type < 1 || type > 2) type = 1;

	m_h_FFC_numElements = (uint32_t)numElements;

	switch (type)
	{
	case 1:
		if (mp_d_FFC_Fluor_Gc != 0)
		{
			cudaError_t err = cudaFree(mp_d_FFC_Fluor_Gc);
		}
		if (mp_d_FFC_Fluor_Dc != 0)
		{
			cudaError_t err = cudaFree(mp_d_FFC_Fluor_Dc);
		}

		cudaMalloc(&mp_d_FFC_Fluor_Gc, numElements * sizeof(float));
		cudaMalloc(&mp_d_FFC_Fluor_Dc, numElements * sizeof(float));

		cudaMemcpy(mp_d_FFC_Fluor_Gc, Gc, numElements * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(mp_d_FFC_Fluor_Dc, Dc, numElements * sizeof(float), cudaMemcpyHostToDevice);
		break;
	case 2:
		if (mp_d_FFC_Lumi_Gc != 0)
		{
			cudaError_t err = cudaFree(mp_d_FFC_Fluor_Gc);
		}
		if (mp_d_FFC_Lumi_Dc != 0)
		{
			cudaError_t err = cudaFree(mp_d_FFC_Fluor_Dc);
		}

		cudaMalloc(&mp_d_FFC_Lumi_Gc, numElements * sizeof(float));
		cudaMalloc(&mp_d_FFC_Lumi_Dc, numElements * sizeof(float));

		cudaMemcpy(mp_d_FFC_Lumi_Gc, Gc, numElements * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(mp_d_FFC_Lumi_Dc, Dc, numElements * sizeof(float), cudaMemcpyHostToDevice);
		break;
	}
}



void CudaImage::FlattenImage(int type)
{
	if (mp_d_grayImage == 0) return; // no image to flatten (a call to SetFullGrayscaleImage or SetRoiGrayscaleImage has not been made)

									 // make sure that the flat field corrector is initialized, if not initialize it so that it has no effect on images
	if (m_h_FFC_numElements != (m_imageW*m_imageH))
	{
		m_h_FFC_numElements = m_imageW*m_imageH;
		float* gc = (float*)malloc(m_imageW*m_imageH * sizeof(float));
		float* dc = (float*)malloc(m_imageW*m_imageH * sizeof(float));
		for (int i = 0; i < m_h_FFC_numElements; i++)
		{
			gc[i] = 1.0;
			dc[i] = 0.0;
		}
		SetFlatFieldCorrectionArrays(1, gc, dc, m_h_FFC_numElements);
		SetFlatFieldCorrectionArrays(2, gc, dc, m_h_FFC_numElements);
	}

	switch (type)
	{
	case 0: // no flattening
		break;
	case 1: // Fluor flattening
		FlattenImage(mp_d_grayImage, mp_d_FFC_Fluor_Gc, mp_d_FFC_Fluor_Dc, m_imageW, m_imageH);
		break;
	case 2: // Lumi flattening
		FlattenImage(mp_d_grayImage, mp_d_FFC_Lumi_Gc, mp_d_FFC_Lumi_Dc, m_imageW, m_imageH);
		break;
	}
}

