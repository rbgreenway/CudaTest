using System;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;




namespace CudaTools
{



    public class ChartArray : IDisposable
    {

        private IntPtr chartArray = IntPtr.Zero;
        const string DLL_NAME = "CudaTools.dll";


        public ChartArray()
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
        ~ChartArray()
        {
            Dispose(false);
        }

        #endregion





        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "InitCudaPlot")]
        //   DllExport bool InitCudaPlot(int chartRows, int chartCols, int chartArrayWidth, int chartArrayHeight,
        //int margin, int padding, int aggregateWidth, int aggregateHeight,
        //uint32_t windowBkgColor,
        //uint32_t chartBkgColor, uint32_t chartSelectedColor, uint32_t chartFrameColor, uint32_t chartAxisColor, uint32_t chartPlotColor,
        //int xmin, int xmax, int ymin, int ymax, int maxNumDataPoints, int numTraces, CudaChartArray** pp_chartArray)
        static extern bool ChartArray_Init(int chartRows, int chartCols, int chartArrayWidth, int chartArrayHeight, int margin, int padding,
                                           int aggregateWidth, int aggregateHeight,
                                           UInt32 windowBkgColor, UInt32 chartBkgColor, UInt32 chartSelectedColor, UInt32 chartFrameColor, UInt32 chartAxisColor, UInt32 chartPlotColor,
                                           int xmin, int xmax, int ymin, int ymax, int maxNumPoints, int numTraces, out IntPtr chartArray);


        [HandleProcessCorruptedStateExceptions]
        [SecurityCritical]
        public bool Init(int chartRows, int chartCols, int chartArrayWidth, int chartArrayHeight, int margin, int padding,
                        int aggregateWidth, int aggregateHeight,
                        Color windowBkgColor, Color chartBkgColor, Color chartSelectedColor, Color chartFrameColor, Color chartAxisColor, Color chartPlotColor,
                        int xmin, int xmax, int ymin, int ymax, int maxNumPoints, int numTraces)
        {
            bool initialized = false;
            chartArray = new IntPtr(0);
            try
            {
                UInt32 col1 = ColorToUInt(windowBkgColor);
                UInt32 col2 = ColorToUInt(chartBkgColor);
                UInt32 col3 = ColorToUInt(chartSelectedColor);
                UInt32 col4 = ColorToUInt(chartFrameColor);
                UInt32 col5 = ColorToUInt(chartAxisColor);
                UInt32 col6 = ColorToUInt(chartPlotColor);

                initialized = ChartArray_Init(chartRows, chartCols, chartArrayWidth, chartArrayHeight, margin, padding,
                    aggregateWidth, aggregateHeight,
                    col1, col2, col3, col4, col5, col6, xmin, xmax, ymin, ymax, maxNumPoints, numTraces, out chartArray);
            }
            catch (Exception ex)
            {
                initialized = false;
                string errMsg = ex.Message;
            }
            return initialized;
        }



        public UInt32 ColorToUInt(Color color)
        {
            return (UInt32)((color.A << 24) | (color.R << 16) |
                          (color.G << 8) | (color.B << 0));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "Shutdown_ChartArray")]
        // void Shutdown_ChartArray(CudaChartArray* p_chart_array)
        static extern void ChartArray_Shutdown(IntPtr pChartArray);

        public void Shutdown()
        {
            try
            {
                ChartArray_Shutdown(chartArray);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetChartArrayPixelSize")]
        //int2 GetChartArrayPixelSize(CudaChartArray* pChartArray)
        static extern UInt64 ChartArray_GetChartArrayPixelSize(IntPtr pChartArray);

        public void GetChartArrayPixelSize(ref int width, ref int height)
        {
            try
            {
                UInt64 val = ChartArray_GetChartArrayPixelSize(chartArray);

                width = (int)(val >> 32);
                height = (int)(val & 0x00000000ffffffff);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "Resize")]
        //  void Resize(CudaChartArray* pChartArray, int chartArrayWidth, int chartArrayHeight, int aggregateWidth, int aggregateHeight)
        static extern void ChartArray_Resize(IntPtr pChartArray, int chartArrayWidth, int chartArrayHeight, int aggregateWidth, int aggregateHeight);

        public void Resize(int chartArrayWidth, int chartArrayHeight, int aggregateWidth, int aggregateHeight)
        {
            try
            {
                ChartArray_Resize(chartArray, chartArrayWidth, chartArrayHeight, aggregateWidth, aggregateHeight);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }



        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetMaxNumberOfTraces")]
        //   int GetMaxNumberOfTraces(CudaChartArray* pChartArray)
        static extern int ChartArray_GetMaxNumberOfTraces(IntPtr pChartArray);

        public int GetMaxNumberOfTraces()
        {
            try
            {
                return ChartArray_GetMaxNumberOfTraces(chartArray);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return 0;
            }
        }





        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetSelectionArrayPtr")]
        // void* GetSelectionArrayPtr(CudaChartArray* pChartArray)
        static extern IntPtr ChartArray_GetSelectionArrayPtr(IntPtr pChartArray);


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "SetSelected")]
        // void SetSelected(CudaChartArray* pChartArray)
        static extern void ChartArray_SetSelectedCharts(IntPtr pChartArray);

        public void SetSelectedCharts(bool[] selectedCharts, ref WriteableBitmap bitmap)
        {
            try
            {
                byte[] sel = new byte[selectedCharts.Length];
                for (int i = 0; i < selectedCharts.Length; i++)
                {
                    if (selectedCharts[i])
                        sel[i] = 1;
                    else
                        sel[i] = 0;
                }

                IntPtr ptr = ChartArray_GetSelectionArrayPtr(chartArray);
                Marshal.Copy(sel, 0, ptr, sel.Length);
                ChartArray_SetSelectedCharts(chartArray);
                Redraw(ref bitmap);

                //GCHandle pinnedArray = GCHandle.Alloc(selectedCharts, GCHandleType.Pinned);
                //IntPtr ptr = pinnedArray.AddrOfPinnedObject();

                //if (ptr != IntPtr.Zero)
                //{
                //    Marshal.Copy(sel, 0, ptr, sel.Length);

                //    ChartArray_SetSelectedCharts(chartArray, ptr);

                //    pinnedArray.Free();

                //    Redraw(ref bitmap);
                //}
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }


        public void SetSelectedCharts(bool[] selectedCharts)
        {
            try
            {
                byte[] sel = new byte[selectedCharts.Length];
                for (int i = 0; i < selectedCharts.Length; i++)
                {
                    if (selectedCharts[i])
                        sel[i] = 1;
                    else
                        sel[i] = 0;
                }

                IntPtr ptr = ChartArray_GetSelectionArrayPtr(chartArray);
                Marshal.Copy(sel, 0, ptr, sel.Length);
                ChartArray_SetSelectedCharts(chartArray);

                //GCHandle pinnedArray = GCHandle.Alloc(selectedCharts, GCHandleType.Pinned);
                //IntPtr ptr = pinnedArray.AddrOfPinnedObject();

                //if (ptr != IntPtr.Zero)
                //{
                //    Marshal.Copy(sel, 0, ptr, sel.Length);

                //    ChartArray_SetSelectedCharts(chartArray, ptr);

                //    pinnedArray.Free();
                //}
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }






        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "SetTraceColor")]
        // void SetTraceColor(CudaChartArray* pChartArray, int traceNum, uint32_t color)
        static extern void ChartArray_SetTraceColor(IntPtr pChartArray, int traceNum, UInt32 color);


        public void SetTraceColor(int traceNum, Color color)
        {
            try
            {
                UInt32 col1 = ColorToUInt(color);

                ChartArray_SetTraceColor(chartArray, traceNum, col1);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }




        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "SetTraceVisibility")]
        // void SetTraceVisibility(CudaChartArray* pChartArray, int traceNum, bool isVisible)
        static extern void ChartArray_SetTraceVisibility(IntPtr pChartArray, int traceNum, bool isVisible);


        public void SetTraceVisibility(int traceNum, bool isVisible)
        {
            try
            {
                ChartArray_SetTraceVisibility(chartArray, traceNum, isVisible);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "SetInitialRanges")]
        // void SetInitialRanges(CudaChartArray* pChartArray, int xmin, int xmax, int ymin, int ymax)
        static extern void ChartArray_SetInitialRanges(IntPtr pChartArray, int xmin, int xmax, int ymin, int ymax);


        public void SetIntialRanges(int xmin, int xmax, int ymin, int ymax)
        {
            try
            {
                ChartArray_SetInitialRanges(chartArray, xmin, xmax, ymin, ymax);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Error Setting Chart Ranges", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }





        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "AppendData")]
        // void AppendData(CudaChartArray* pChartArray, int* xArray, int* yArray, int numPoints, int traceNum)
        static extern void ChartArray_AppendData(IntPtr pChartArray, IntPtr xArray, IntPtr yArray, int numPoints, int traceNum);

        public void AppendData(int[] xVals, int[] yVals, int traceNum, ref WriteableBitmap bitmap)
        {

            try
            {
                // add new points to charts, update the charts, and redraw to screen

                GCHandle pinnedArray1 = GCHandle.Alloc(xVals, GCHandleType.Pinned);
                IntPtr ptr1 = pinnedArray1.AddrOfPinnedObject();
                Marshal.Copy(xVals, 0, ptr1, xVals.Length);

                GCHandle pinnedArray2 = GCHandle.Alloc(yVals, GCHandleType.Pinned);
                IntPtr ptr2 = pinnedArray2.AddrOfPinnedObject();
                Marshal.Copy(yVals, 0, ptr2, yVals.Length);

                ChartArray_AppendData(chartArray, ptr1, ptr2, xVals.Length, traceNum);

                IntPtr ptr = ChartArray_GetChartImagePtr(chartArray);

                if (ptr != IntPtr.Zero)
                {
                    int width = (int)bitmap.Width;
                    int height = (int)bitmap.Height;

                    System.Windows.Int32Rect rect = new System.Windows.Int32Rect(0, 0, width, height);

                    bitmap.Lock();
                    bitmap.WritePixels(rect, ptr, width * height * 4, width * 4);
                    bitmap.Unlock();
                }

                pinnedArray1.Free();
                pinnedArray2.Free();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }


        public void AppendData(int[] xVals, int[] yVals, int traceNum)
        {
            try
            {
                // add points to charts without copying the new charts back to the display bitmap

                GCHandle pinnedArray1 = GCHandle.Alloc(xVals, GCHandleType.Pinned);
                IntPtr ptr1 = pinnedArray1.AddrOfPinnedObject();
                Marshal.Copy(xVals, 0, ptr1, xVals.Length);

                GCHandle pinnedArray2 = GCHandle.Alloc(yVals, GCHandleType.Pinned);
                IntPtr ptr2 = pinnedArray2.AddrOfPinnedObject();
                Marshal.Copy(yVals, 0, ptr2, yVals.Length);

                ChartArray_AppendData(chartArray, ptr1, ptr2, xVals.Length, traceNum);

                pinnedArray1.Free();
                pinnedArray2.Free();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }


        public void Refresh(ref WriteableBitmap bitmap)
        {
            try
            {
                // refresh the display with the current image

                IntPtr ptr = ChartArray_GetChartImagePtr(chartArray);

                if (ptr != IntPtr.Zero)
                {
                    int width = (int)bitmap.Width;
                    int height = (int)bitmap.Height;

                    System.Windows.Int32Rect rect = new System.Windows.Int32Rect(0, 0, width, height);

                    bitmap.Lock();
                    bitmap.WritePixels(rect, ptr, width * height * 4, width * 4);
                    bitmap.Unlock();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public void RefreshAggregate(ref WriteableBitmap bitmap)
        {
            try
            {
                // refresh the display with the current image

                IntPtr ptr = ChartArray_GetAggregateImagePtr(chartArray);

                if (ptr != IntPtr.Zero)
                {
                    int width = (int)bitmap.Width;
                    int height = (int)bitmap.Height;

                    System.Windows.Int32Rect rect = new System.Windows.Int32Rect(0, 0, width, height);

                    bitmap.Lock();
                    bitmap.WritePixels(rect, ptr, width * height * 4, width * 4);
                    bitmap.Unlock();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "Redraw")]
        // void Redraw(CudaChartArray* pChartArray)
        static extern void ChartArray_Redraw(IntPtr pChartArray);

        public void Redraw(ref WriteableBitmap bitmap)
        {
            try
            {
                ChartArray_Redraw(chartArray);

                IntPtr ptr = ChartArray_GetChartImagePtr(chartArray);

                if (ptr != IntPtr.Zero)
                {
                    int width = (int)bitmap.Width;
                    int height = (int)bitmap.Height;

                    System.Windows.Int32Rect rect = new System.Windows.Int32Rect(0, 0, width, height);

                    bitmap.Lock();
                    bitmap.WritePixels(rect, ptr, width * height * 4, width * 4);
                    bitmap.Unlock();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

        }

        public void Redraw()
        {
            try
            {
                // redraws into memory without updating display
                ChartArray_Redraw(chartArray);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "RedrawAggregate")]
        // void RedrawAggregate(CudaChartArray* pChartArray)
        static extern void ChartArray_RedrawAggregate(IntPtr pChartArray);

        public void RedrawAggregate(ref WriteableBitmap bitmap)
        {
            try
            {
                ChartArray_Redraw(chartArray);

                IntPtr ptr = ChartArray_GetAggregateImagePtr(chartArray);

                if (ptr != IntPtr.Zero)
                {
                    int width = (int)bitmap.Width;
                    int height = (int)bitmap.Height;

                    System.Windows.Int32Rect rect = new System.Windows.Int32Rect(0, 0, width, height);

                    bitmap.Lock();
                    bitmap.WritePixels(rect, ptr, width * height * 4, width * 4);
                    bitmap.Unlock();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

        }

        public void RedrawAggregate()
        {
            try
            {
                // redraws into memory without updating display
                ChartArray_RedrawAggregate(chartArray);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetChartImagePtr")]
        // void* GetChartImagePtr(CudaChartArray* pChartArray)
        static extern IntPtr ChartArray_GetChartImagePtr(IntPtr pChartArray);

        public IntPtr GetImagePtr()
        {
            try
            {
                return ChartArray_GetChartImagePtr(chartArray);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return IntPtr.Zero;
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetAggregateImagePtr")]
        // void* GetAggregateImagePtr(CudaChartArray* pChartArray)
        static extern IntPtr ChartArray_GetAggregateImagePtr(IntPtr pChartArray);

        public IntPtr GetAggregateImagePtr()
        {
            try
            {
                return ChartArray_GetAggregateImagePtr(chartArray);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return IntPtr.Zero;
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetRangePtr")]
        // void* GetRangePtr(CudaChartArray* pChartArray)
        static extern IntPtr ChartArray_GetRangePtr(IntPtr pChartArray);

        public void GetRanges(ref int[] range)
        {
            try
            {
                IntPtr rangePtr = ChartArray_GetRangePtr(chartArray);

                if (rangePtr != IntPtr.Zero)
                {
                    Marshal.Copy(rangePtr, range, 0, 4);
                }

                rangePtr = IntPtr.Zero;
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetRowFromY")]
        // int32_t GetRowFromY(CudaChartArray* pChartArray, int32_t y)
        static extern Int32 ChartArray_GetRowFromY(IntPtr pChartArray, Int32 y);

        public Int32 GetRowFromY(Int32 y)
        {
            try
            {
                return ChartArray_GetRowFromY(chartArray, y);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return 0;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetColumnFromX")]
        // int32_t GetColumnFromX(CudaChartArray* pChartArray, int32_t x)
        static extern Int32 ChartArray_GetColumnFromX(IntPtr pChartArray, Int32 x);

        public Int32 GetColumnFromX(Int32 x)
        {
            try
            {
                return ChartArray_GetColumnFromX(chartArray, x);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return 0;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl, EntryPoint = "Reset")]
        // void Reset(CudaChartArray* pChartArray)
        static extern void ChartArray_Reset(IntPtr pChartArray);

        public void Reset()
        {
            try
            {
                ChartArray_Reset(chartArray);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Marshaling Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }



    }



}