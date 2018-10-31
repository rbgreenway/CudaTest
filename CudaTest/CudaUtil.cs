using System;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Windows;

namespace CudaTools
{


    public class CudaUtil : IDisposable
    {
        private IntPtr cudaUtil = IntPtr.Zero;
        const string DLL_NAME = "CudaTools.dll";

        // Constructor
        public CudaUtil()
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
        ~CudaUtil()
        {
            Dispose(false);
        }

        #endregion





        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "InitCudaUtil")]
        //  void InitCudaUtil(CudaUtil** pp_CudaUtil)
        static extern bool CudaUtil_Init(out IntPtr cudaUtil);

        [HandleProcessCorruptedStateExceptions]
        [SecurityCritical]
        public bool Init()
        {
            bool initialized = false;
            cudaUtil = new IntPtr(0);
            try
            {
                CudaUtil_Init(out cudaUtil);
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

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "Shutdown_CudaUtil")]
        // void Shutdown_CudaUtil(CudaUtil* pCudaUtil)
        static extern void CudaUtil_Shutdown(IntPtr pCudaUtil);

        public void Shutdown()
        {
            try
            {
                CudaUtil_Shutdown(cudaUtil);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////



        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetCudaDeviceCount")]
        // int GetCudaDeviceCount(CudaUtil* pCudaUtil)
        static extern int CudaUtil_GetCudaDeviceCount(IntPtr pCudaUtil);

        public int GetCudaDeviceCount()
        {
            int count = 0;
            try
            {
                count = CudaUtil_GetCudaDeviceCount(cudaUtil);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            return count;
        }



        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetCudaComputeCapability")]
        // void GetCudaComputeCapability(CudaUtil* pCudaUtil, int* major, int* minor)
        static extern void CudaUtil_GetCudaComputeCapability(IntPtr pCudaUtil, IntPtr pMajor, IntPtr pMinor);

        public int GetCudaDeviceComputeCapability(ref int major, ref int minor)
        {
            int count = 0;
            try
            {
                // Allocate unmanaged memory.                 
                IntPtr pMajor = Marshal.AllocHGlobal(sizeof(int));
                IntPtr pMinor = Marshal.AllocHGlobal(sizeof(int));

                CudaUtil_GetCudaComputeCapability(cudaUtil, pMajor, pMinor);

                major = Marshal.ReadInt32(pMajor);
                minor = Marshal.ReadInt32(pMinor);

                Marshal.FreeHGlobal(pMajor);
                Marshal.FreeHGlobal(pMinor);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            return count;
        }





        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetCudaDeviceName")]
        //void GetCudaDeviceName(CudaUtil* pCudaUtil, char* pName, int* pLen)
        static extern void CudaUtil_GetCudaDeviceName(IntPtr pCudaUtil, IntPtr pName, IntPtr pLength);

        public string GetCudaDeviceName()
        {
            string name = "";
            try
            {
                // Allocate unmanaged memory.                 
                IntPtr pName = Marshal.AllocHGlobal(256);
                IntPtr pLength = Marshal.AllocHGlobal(sizeof(int));

                CudaUtil_GetCudaDeviceName(cudaUtil, pName, pLength);

                int len = Marshal.ReadInt32(pLength);
                char[] deviceName = new char[len];
                Array.Clear(deviceName, 0, len);

                for (int i = 0; i < len; i++)
                    deviceName[i] = (char)Marshal.ReadByte(pName, i);

                name = new string(deviceName);


                Marshal.FreeHGlobal(pName);
                Marshal.FreeHGlobal(pLength);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            return name;
        }




        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetCudaLastErrorMessage")]
        // void GetCudaLastErrorMessage(CudaUtil* pCudaUtil, char* pMessage, int* pLen)
        static extern void CudaUtil_GetCudaLastError(IntPtr pCudaUtil, IntPtr pMessage, IntPtr pLength);

        public string GetCudaLastError()
        {
            string msg = "";
            try
            {
                // Allocate unmanaged memory.                 
                IntPtr pMessage = Marshal.AllocHGlobal(256);
                IntPtr pLength = Marshal.AllocHGlobal(sizeof(int));

                CudaUtil_GetCudaLastError(cudaUtil, pMessage, pLength);

                int len = Marshal.ReadInt32(pLength);
                char[] errorMsg = new char[len];
                Array.Clear(errorMsg, 0, len);

                for (int i = 0; i < len; i++)
                    errorMsg[i] = (char)Marshal.ReadByte(pMessage, i);

                msg = new string(errorMsg);


                Marshal.FreeHGlobal(pMessage);
                Marshal.FreeHGlobal(pLength);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            return msg;
        }



        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetCudaDeviceMemory")]
        // void GetCudaDeviceMemory(CudaUtil* pCudaUtil, size_t* pTotMem, size_t* pFreeMem)
        static extern void CudaUtil_GetCudaDeviceMemory(IntPtr pCudaUtil, IntPtr pTotal, IntPtr pFree);

        public void GetCudaDeviceMemory(ref Int64 totalMem, ref Int64 freeMem)
        {
            try
            {
                // Allocate unmanaged memory.                 
                IntPtr pTotalMem = Marshal.AllocHGlobal(sizeof(Int64));
                IntPtr pFreeMem = Marshal.AllocHGlobal(sizeof(Int64));

                CudaUtil_GetCudaDeviceMemory(cudaUtil, pTotalMem, pFreeMem);

                totalMem = Marshal.ReadInt64(pTotalMem);
                freeMem = Marshal.ReadInt64(pFreeMem);

                Marshal.FreeHGlobal(pTotalMem);
                Marshal.FreeHGlobal(pFreeMem);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }





    }

}
