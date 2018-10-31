using System;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Windows;




namespace CudaTools
{
    public class ImageTool : IDisposable
    {
        private IntPtr imageTool = IntPtr.Zero;
        const string DLL_NAME = "CudaTools.dll";

        // Constructor
        public ImageTool()
        {
        }

        #region Resource Disposable

        // Flag: Has Dispose already been called?
        bool disposed = false;

        // Public implementation of Dispose pattern callable by consumers.
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        // Protected implementation of Dispose pattern.
        [HandleProcessCorruptedStateExceptions]
        [SecurityCriticalAttribute]
        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    // Free any other managed objects here.
                }

                // Free any unmanaged objects here.
                try
                {
                    Shutdown();
                }
                catch (Exception e)
                {
                    // Catch any unmanaged exceptions
                }
                disposed = true;
            }

        }

        // Destructor (.NET Finalize)
        ~ImageTool()
        {
            Dispose(false);
        }

        #endregion





        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "InitImageTool")]
        //  void InitImageTool(CudaImage** pp_CudaImage)
        static extern bool ImageTool_Init(out IntPtr imageTool);

        [HandleProcessCorruptedStateExceptions]
        [SecurityCritical]
        public bool Init()
        {
            bool initialized = false;
            imageTool = new IntPtr(0);
            try
            {
                ImageTool_Init(out imageTool);
                initialized = true;
            }
            catch (Exception ex)
            {
                initialized = false;
                string errMsg = ex.Message;
            }
            return initialized;
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "Shutdown_ImageTool")]
        //void Shutdown_ImageTool(CudaImage* pCudaImage)
        static extern void ImageTool_Shutdown(IntPtr pImageTool);

        public void Shutdown()
        {
            try
            {
                ImageTool_Shutdown(imageTool);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////



        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "SetFullGrayscaleImage")]
        //uint16_t* SetFullGrayscaleImage(CudaImage* pCudaImage, uint16_t* grayImage, uint16_t imageWidth, uint16_t imageHeight)
        static extern IntPtr ImageTool_SetFullGrayscaleImage(IntPtr pImageTool, IntPtr grayImage, UInt16 imageWidth, UInt16 imageHeight);

        public IntPtr PostFullGrayscaleImage(UInt16[] grayImage, UInt16 width, UInt16 height)
        {
            // copy 16-bit grayscale image to GPU
            GCHandle pinnedArray = GCHandle.Alloc(grayImage, GCHandleType.Pinned);
            IntPtr grayImagePointer = pinnedArray.AddrOfPinnedObject();

            IntPtr returnPtr = IntPtr.Zero;
            returnPtr = ImageTool_SetFullGrayscaleImage(imageTool, grayImagePointer, width, height);

            pinnedArray.Free();

            return returnPtr;  // returns pointer to GPU memory where the gray image resides
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "SetRoiGrayscaleImage")]
        //uint16_t* SetRoiGrayscaleImage(CudaImage* pCudaImage, uint16_t* roiImage, uint16_t imageWidth, uint16_t imageHeight,
        //                               uint16_t roiWidth, uint16_t roiHeight, uint16_t roiX, uint16_t roiY)
        static extern IntPtr ImageTool_SetRoiGrayscaleImage(IntPtr pImageTool, IntPtr roiImage, UInt16 imageWidth, UInt16 imageHeight,
                                         UInt16 roiWidth, UInt16 roiHeight, UInt16 roiX, UInt16 roiY);

        public IntPtr PostRoiGrayscaleImage(UInt16[] roiImage, UInt16 width, UInt16 height, UInt16 roiWidth, UInt16 roiHeight, UInt16 roiX, UInt16 roiY)
        {
            // copy 16-bit roi grayscale image to GPU
            GCHandle pinnedArray = GCHandle.Alloc(roiImage, GCHandleType.Pinned);
            IntPtr roiImagePointer = pinnedArray.AddrOfPinnedObject();

            IntPtr returnPtr = IntPtr.Zero;
            returnPtr = ImageTool_SetRoiGrayscaleImage(imageTool, roiImagePointer, width, height, roiWidth, roiHeight, roiX, roiY);

            pinnedArray.Free();

            return returnPtr;  // returns pointer to GPU memory where the full (not just the roi) gray image resides
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetGrayscaleImagePtr")]
        //uint16_t* GetGrayscaleImagePtr(CudaImage* pCudaImage)
        static extern IntPtr ImageTool_GetGrayscaleImagePtr(IntPtr pImageTool);

        public IntPtr GetGrayscaleImagePtr()
        {
            return ImageTool_GetGrayscaleImagePtr(imageTool);
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "SetMaskImage")]
        //uint16_t* SetMaskImage(CudaImage* pCudaImage, uint16_t* maskImage, uint16_t maskWidth, uint16_t maskHeight, uint16_t maskRows, uint16_t maskCols)
        static extern IntPtr ImageTool_SetMaskImage(IntPtr pImageTool, IntPtr maskImage, UInt16 maskWidth, UInt16 maskHeight, UInt16 maskRows, UInt16 maskCols);


        public IntPtr Set_MaskImage(UInt16[] maskImage, UInt16 maskWidth, UInt16 maskHeight, UInt16 maskRows, UInt16 maskCols)
        {
            // copy 16-bit image mask to GPU

            IntPtr returnPtr = IntPtr.Zero;

            // Initialize unmanaged memory to hold the array.
            int size = Marshal.SizeOf(maskImage[0]) * maskImage.Length;
            IntPtr pnt = Marshal.AllocHGlobal(size);

            try
            {
                // Copy the array to unmanaged memory.
                Marshal.Copy((Int16[])((object)maskImage), 0, pnt, maskImage.Length);  // 

                // copy unmanaged array to GPU
                returnPtr = ImageTool_SetMaskImage(imageTool, pnt, maskWidth, maskHeight, maskRows, maskCols);
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pnt);
            }

            return returnPtr;  // returns pointer to GPU memory where the mask image resides
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetMaskImagePtr")]
        // uint16_t* GetMaskImagePtr(CudaImage* pCudaImage)
        static extern IntPtr ImageTool_GetMaskImagePtr(IntPtr pImageTool);

        public IntPtr GetMaskImagePtr()
        {
            return ImageTool_GetMaskImagePtr(imageTool);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "SetColorMap")]
        // void SetColorMap(CudaImage* pCudaImage, uint8_t* redMap, uint8_t* greenMap, uint8_t* blueMap, uint16_t maxPixelValue)
        static extern void ImageTool_SetColorMap(IntPtr pImageTool, IntPtr redMap, IntPtr greenMap, IntPtr blueMap, UInt16 maxPixelValue);

        public void Set_ColorMap(byte[] redMap, byte[] greenMap, byte[] blueMap, UInt16 maxPixelValue)
        {
            // This funciton sets the color maps used for each color onto the GPU

            // Initialize unmanaged memory to hold the arrays
            int sizeRed = Marshal.SizeOf(redMap[0]) * redMap.Length;
            IntPtr pntRed = Marshal.AllocHGlobal(sizeRed);

            int sizeGreen = Marshal.SizeOf(greenMap[0]) * greenMap.Length;
            IntPtr pntGreen = Marshal.AllocHGlobal(sizeGreen);

            int sizeBlue = Marshal.SizeOf(blueMap[0]) * blueMap.Length;
            IntPtr pntBlue = Marshal.AllocHGlobal(sizeBlue);

            try
            {
                // Copy the array to unmanaged memory.
                Marshal.Copy(redMap, 0, pntRed, redMap.Length);
                Marshal.Copy(greenMap, 0, pntGreen, greenMap.Length);
                Marshal.Copy(blueMap, 0, pntBlue, blueMap.Length);

                // copy color maps from unmanaged memory to GPU memory
                ImageTool_SetColorMap(imageTool, pntRed, pntGreen, pntBlue, maxPixelValue);
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pntRed);
                Marshal.FreeHGlobal(pntGreen);
                Marshal.FreeHGlobal(pntBlue);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ConvertGrayscaleToColor")]
        // uint8_t* ConvertGrayscaleToColor(CudaImage* pCudaImage, uint16_t scaleLower, uint16_t scaleUpper)
        static extern IntPtr ImageTool_ConvertGrayscaleToColor(IntPtr pImageTool, UInt16 scaleLower, UInt16 scaleUpper);

        public IntPtr Convert_GrayscaleToColor(UInt16 scaleLower, UInt16 scaleUpper)
        {
            IntPtr ptr = IntPtr.Zero;
            ptr = ImageTool_ConvertGrayscaleToColor(imageTool, scaleLower, scaleUpper);
            return ptr;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetColorImagePtr")]
        // uint8_t* GetColorImagePtr(CudaImage* pCudaImage)
        static extern IntPtr ImageTool_GetColorImagePtr(IntPtr pImageTool);

        public IntPtr GetColorImagePtr()
        {
            return ImageTool_GetColorImagePtr(imageTool);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ApplyMaskToImage")]
        // void ApplyMaskToImage(CudaImage* pCudaImage)
        static extern void ImageTool_ApplyMaskToImage(IntPtr pImageTool);

        public void ApplyMaskToGrayscaleImage()
        {
            ImageTool_ApplyMaskToImage(imageTool);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "SetFlatFieldCorrectionArrays")]
        // void SetFlatFieldCorrectionArrays(CudaImage* pCudaImage, int type, float* Gc, float* Dc, int numElements)
        static extern void ImageTool_SetFlatFieldCorrectionArrays(IntPtr pImageTool, int type, IntPtr Gc, IntPtr Dc, int numElements);

        public void SetFlatFieldCorrection(int type, float[] Gc, UInt16[] Dc)
        {
            // Initialize unmanaged memory to hold the arrays
            int sizeGc = Marshal.SizeOf(Gc[0]) * Gc.Length;
            IntPtr pntGc = Marshal.AllocHGlobal(sizeGc);

            // have to copy Dc to an float[], since Marshal.Copy doesn't accept a UInt16[] as an argument
            float[] floatDc = new float[Dc.Length];
            for (int i = 0; i < Dc.Length; i++) floatDc[i] = (float)Dc[i];
            int sizeDc = Marshal.SizeOf(floatDc[0]) * floatDc.Length;
            IntPtr pntDc = Marshal.AllocHGlobal(sizeDc);

            try
            {
                // Copy the array to unmanaged memory.
                Marshal.Copy(Gc, 0, pntGc, Gc.Length);
                Marshal.Copy(floatDc, 0, pntDc, floatDc.Length);

                // Call into unmanaged DLL
                ImageTool_SetFlatFieldCorrectionArrays(imageTool, type, pntGc, pntDc, Gc.Length);

            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pntGc);
                Marshal.FreeHGlobal(pntDc);
            }

        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "FlattenImage")]
        // void void FlattenImage(CudaImage* pCudaImage, int type)
        static extern void ImageTool_FlattenImage(IntPtr pImageTool, int type);

        public void FlattenGrayImage(int type)
        {
            ImageTool_FlattenImage(imageTool, type);
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "DownloadColorImage")]
        // void DownloadColorImage(CudaImage* pCudaImage, uint8_t* colorImageDest)
        static extern void ImageTool_DownloadColorImage(IntPtr pImageTool, IntPtr colorImageDest);

        public void Download_ColorImage(out byte[] colorImage, UInt16 width, UInt16 height)
        {
            colorImage = new byte[width * height * 4];
            GCHandle pinnedArray = GCHandle.Alloc(colorImage, GCHandleType.Pinned);
            IntPtr ptr = pinnedArray.AddrOfPinnedObject();

            ImageTool_DownloadColorImage(imageTool, ptr);

            pinnedArray.Free();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "DownloadGrayscaleImage")]
        // void DownloadGrayscaleImage(CudaImage* pCudaImage, uint16_t* grayImageDest)
        static extern void ImageTool_DownloadGrayscaleImage(IntPtr pImageTool, IntPtr grayscaleImageDest);

        public void Download_GrayscaleImage(out UInt16[] grayscaleImage, UInt16 width, UInt16 height)
        {
            grayscaleImage = new UInt16[width * height];
            GCHandle pinnedArray = GCHandle.Alloc(grayscaleImage, GCHandleType.Pinned);
            IntPtr ptr = pinnedArray.AddrOfPinnedObject();

            ImageTool_DownloadGrayscaleImage(imageTool, ptr);

            pinnedArray.Free();
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetHistogram_512Buckets")]
        // void GetHistogram_512Buckets(CudaImage* pCudaImage, uint32_t* destHist, uint8_t maxValueBitWidth)
        static extern void ImageTool_GetHistogram_512Buckets(IntPtr pImageTool, IntPtr destHist, byte maxValueBitWidth);

        public void GetHistogram_512(out UInt32[] histogram, byte maxValueBitWidth)
        {
            histogram = new UInt32[512];
            GCHandle pinnedArray = GCHandle.Alloc(histogram, GCHandleType.Pinned);
            IntPtr ptr = pinnedArray.AddrOfPinnedObject();

            ImageTool_GetHistogram_512Buckets(imageTool, ptr, maxValueBitWidth);

            pinnedArray.Free();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetHistogramImage_512Buckets")]
        // void GetHistogramImage_512Buckets(CudaImage* pCudaImage, uint8_t* histImage, uint16_t width, uint16_t height, uint32_t maxBinCount)
        static extern void ImageTool_GetHistogramImage_512Buckets(IntPtr pImageTool, IntPtr histImage, UInt16 width, UInt16 height, UInt32 maxBinCount);

        public void GetHistogramImage_512(out byte[] histImage, UInt16 width, UInt16 height, UInt32 maxBinCount)
        {
            histImage = new byte[width * height * 4];
            GCHandle pinnedArray = GCHandle.Alloc(histImage, GCHandleType.Pinned);
            IntPtr ptr = pinnedArray.AddrOfPinnedObject();

            ImageTool_GetHistogramImage_512Buckets(imageTool, ptr, width, height, maxBinCount);

            pinnedArray.Free();
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "CalculateMaskApertureSums")]
        // void CalculateMaskApertureSums(CudaImage* pCudaImage, uint32_t* sums)
        static extern void ImageTool_CalculateMaskApertureSums(IntPtr pImageTool, IntPtr sums);

        public void GetMaskApertureSums(out UInt32[] sums, int maskApertureRows, int maskApertureCols)
        {
            sums = new UInt32[maskApertureRows * maskApertureCols];
            GCHandle pinnedArray = GCHandle.Alloc(sums, GCHandleType.Pinned);
            IntPtr ptr = pinnedArray.AddrOfPinnedObject();

            ImageTool_CalculateMaskApertureSums(imageTool, ptr);

            pinnedArray.Free();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetImageAverage")]
        // void GetImageAverage(CudaImage* pCudaImage, uint16_t* grayImage, int width, int height, uint16_t* pAverage)
        static extern void ImageTool_GetImageAverage(IntPtr pImageTool, IntPtr grayImage, UInt16 imageWidth, UInt16 imageHeight, IntPtr average);

        public UInt16 GetImageAverage(UInt16[] grayImage, UInt16 width, UInt16 height)
        {
            // copy 16-bit grayscale image to GPU
            GCHandle pinnedArray = GCHandle.Alloc(grayImage, GCHandleType.Pinned);
            IntPtr grayImagePointer = pinnedArray.AddrOfPinnedObject();

            int average = 0;
            GCHandle pinnedAverage = GCHandle.Alloc(average, GCHandleType.Pinned);
            IntPtr averagePointer = pinnedAverage.AddrOfPinnedObject();

            ImageTool_GetImageAverage(imageTool, grayImagePointer, width, height, averagePointer);

            average = Marshal.ReadInt32(averagePointer, 0);

            pinnedArray.Free();
            pinnedAverage.Free();

            return (UInt16)average;
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////



        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetGrayImageAverage")]
        // DllExport void GetGrayImageAverage(CudaImage* pCudaImage, int* pAverage)
        static extern void ImageTool_GetGrayImageAverage(IntPtr pImageTool, IntPtr average);

        public UInt16 GetGrayImageAverage()
        {
            int average = 0;
            GCHandle pinnedAverage = GCHandle.Alloc(average, GCHandleType.Pinned);
            IntPtr averagePointer = pinnedAverage.AddrOfPinnedObject();

            ImageTool_GetGrayImageAverage(imageTool, averagePointer);

            average = Marshal.ReadInt32(averagePointer, 0);

            pinnedAverage.Free();

            return (UInt16)average;
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    }

}