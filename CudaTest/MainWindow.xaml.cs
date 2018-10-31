using System;
using System.ComponentModel;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Threading;
using System.Collections.Generic;

namespace CudaTest
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {     
        MainWindow_ViewModel m_vm; 

        int m_x = 0;
        int m_y = 0;
  
        Stopwatch m_sw;

        int[] m_dummyData;
        int m_dummyDataSize;

        long m_duration;
        
        public MainWindow()
        {
            InitializeComponent();

            m_vm = new MainWindow_ViewModel();
            DataContext = m_vm;

            CudaTools.CudaUtil cudaUtil = new CudaTools.CudaUtil();
            cudaUtil.Init();

            int numCudaDevices = cudaUtil.GetCudaDeviceCount();
            int major=0, minor=0;
            cudaUtil.GetCudaDeviceComputeCapability(ref major, ref minor);
            string name = cudaUtil.GetCudaDeviceName();

            Int64 totalMem = 0, freeMem = 0;
            cudaUtil.GetCudaDeviceMemory(ref totalMem, ref freeMem);

            CudaTools.ImageTool imageTool = new CudaTools.ImageTool();
            imageTool.Init();


            List<WPFTools.MultiChartArray_TraceItem> traces = new List<WPFTools.MultiChartArray_TraceItem>();
            traces.Add(new WPFTools.MultiChartArray_TraceItem("Indicator 1", 0, 100));
            traces.Add(new WPFTools.MultiChartArray_TraceItem("Indicator 2", 1, 200));

            ChartArray.Init(16, 24, 10000, traces);  // NOTE: allocate for expected max number of points
                                                     // ~ 120 Megabytes/trace of GPU memory required for every 10,000 points

            ChartArray.SetDefaultChartRanges(WPFTools.MultiChartArray.SIGNAL_TYPE.RAW, 0, 10, 0, 10);
            ChartArray.SetDefaultChartRanges(WPFTools.MultiChartArray.SIGNAL_TYPE.CONTROL_SUBTRACTION, 0, 20, 0, 20);
            ChartArray.SetDefaultChartRanges(WPFTools.MultiChartArray.SIGNAL_TYPE.STATIC_RATIO, 0, 30, 0, 30);
            ChartArray.SetDefaultChartRanges(WPFTools.MultiChartArray.SIGNAL_TYPE.DYNAMIC_RATIO, 0, 40, 0, 40);

            ChartArray.SetTraceColor(WPFTools.MultiChartArray.SIGNAL_TYPE.RAW, 0, System.Windows.Media.Colors.Yellow);
            ChartArray.SetTraceColor(WPFTools.MultiChartArray.SIGNAL_TYPE.CONTROL_SUBTRACTION, 0, System.Windows.Media.Colors.OrangeRed);
            ChartArray.SetTraceColor(WPFTools.MultiChartArray.SIGNAL_TYPE.STATIC_RATIO, 0, System.Windows.Media.Colors.CornflowerBlue);
            ChartArray.SetTraceColor(WPFTools.MultiChartArray.SIGNAL_TYPE.DYNAMIC_RATIO, 0, System.Windows.Media.Colors.LawnGreen);

            ChartArray.SetTraceColor(WPFTools.MultiChartArray.SIGNAL_TYPE.RAW, 1, System.Windows.Media.Colors.Purple);
            ChartArray.SetTraceColor(WPFTools.MultiChartArray.SIGNAL_TYPE.CONTROL_SUBTRACTION, 1, System.Windows.Media.Colors.Plum);
            ChartArray.SetTraceColor(WPFTools.MultiChartArray.SIGNAL_TYPE.STATIC_RATIO, 1, System.Windows.Media.Colors.LimeGreen);
            ChartArray.SetTraceColor(WPFTools.MultiChartArray.SIGNAL_TYPE.DYNAMIC_RATIO, 1, System.Windows.Media.Colors.Goldenrod);

            m_duration = 0;

            m_dummyDataSize = 50000;
            m_dummyData = new int[m_dummyDataSize];

            for(int i = 0; i<m_dummyDataSize; i++)
            {                
                double angle = ((double)i) / 80.0;                
                m_dummyData[i] = (int)((100.0+(double)i) * Math.Sin(angle));
            }



            int w = 1024;
            int h = 1024;
            UInt16 val = 654;

            UInt16[] image = SynthesizeGrayImage(w, h, val);

            imageTool.PostFullGrayscaleImage(image, (UInt16)w, (UInt16)h);


            Stopwatch sw = new Stopwatch();
            
            sw.Start();
            //UInt16 avg = imageTool.GetImageAverage(image, (UInt16)w, (UInt16)h);
            UInt16 avg = imageTool.GetGrayImageAverage();
            sw.Stop();
            long t = sw.ElapsedMilliseconds;
            long t1 = sw.ElapsedTicks;
            double usecs = ((double)t1 / (double)Stopwatch.Frequency) * 1000000.0;

            MessageBox.Show("Average = " + avg.ToString() + "     Execution Time = " + usecs.ToString() + " usecs", 
                "Information", MessageBoxButton.OK, MessageBoxImage.Information);

        }


        private void BackgroundTask(int count, int delay)
        {  
            Dispatcher.BeginInvoke(new Action(() =>
            {
                StartPB.IsEnabled = false;
                ResetPB.IsEnabled = false;
                RunningIndicator.IsRunning(true);
            }), DispatcherPriority.Background);
            

            int num = 0;
            long[] intervals = new long[count];
            long last = 0;
            bool flip = false;

            TimeSpan delayTime = TimeSpan.FromTicks(TimeSpan.TicksPerMillisecond / 2);

            int[] x = new int[ChartArray.NumRows() * ChartArray.NumCols()];

            int[] y1 = new int[ChartArray.NumRows() * ChartArray.NumCols()];
            int[] y2 = new int[ChartArray.NumRows() * ChartArray.NumCols()];
            int[] y3 = new int[ChartArray.NumRows() * ChartArray.NumCols()];
            int[] y4 = new int[ChartArray.NumRows() * ChartArray.NumCols()];

            int[] y5 = new int[ChartArray.NumRows() * ChartArray.NumCols()];
            int[] y6 = new int[ChartArray.NumRows() * ChartArray.NumCols()];
            int[] y7 = new int[ChartArray.NumRows() * ChartArray.NumCols()];
            int[] y8 = new int[ChartArray.NumRows() * ChartArray.NumCols()];


            Stopwatch sw = new Stopwatch();
            sw.Start();
            while (num<count)
            {
                while (sw.ElapsedMilliseconds - last < delay)
                {
                    Thread.Sleep(delayTime);
                }
                intervals[num] = sw.ElapsedMilliseconds - last;
                last = sw.ElapsedMilliseconds;
                num++;

                for (int i = 0; i < ChartArray.NumRows() * ChartArray.NumCols(); i++)
                {
                    x[i] = m_x;

                    y1[i] = m_dummyData[num+i];
                    y2[i] = 2*m_dummyData[num+i+50];
                    y3[i] = 3*m_dummyData[num+i+75];
                    y4[i] = 4*m_dummyData[num+i+200];

                    y5[i] = m_dummyData[num + i + 300];
                    y6[i] = 2 * m_dummyData[num + i + 450];
                    y7[i] = 3 * m_dummyData[num + i + 675];
                    y8[i] = 4 * m_dummyData[num + i + 600];
                }

                if (flip)
                {
                    ChartArray.AppendData(x, y1, WPFTools.MultiChartArray.SIGNAL_TYPE.RAW, 100);
                    ChartArray.AppendData(x, y2, WPFTools.MultiChartArray.SIGNAL_TYPE.CONTROL_SUBTRACTION, 100);
                    ChartArray.AppendData(x, y3, WPFTools.MultiChartArray.SIGNAL_TYPE.STATIC_RATIO, 100);
                    ChartArray.AppendData(x, y4, WPFTools.MultiChartArray.SIGNAL_TYPE.DYNAMIC_RATIO, 100);
                }
                else
                {
                    ChartArray.AppendData(x, y5, WPFTools.MultiChartArray.SIGNAL_TYPE.RAW, 200);
                    ChartArray.AppendData(x, y6, WPFTools.MultiChartArray.SIGNAL_TYPE.CONTROL_SUBTRACTION, 200);
                    ChartArray.AppendData(x, y7, WPFTools.MultiChartArray.SIGNAL_TYPE.STATIC_RATIO, 200);
                    ChartArray.AppendData(x, y8, WPFTools.MultiChartArray.SIGNAL_TYPE.DYNAMIC_RATIO, 200);
                }

                flip = !flip;
                m_x++;
                m_y = m_x;
            }

   
            m_duration = sw.ElapsedMilliseconds;
            
            //Debug.Print("total time = " + sw.ElapsedMilliseconds.ToString() + "   for  " + count.ToString() + "  Points");

            Dispatcher.BeginInvoke(new Action(() =>
            {
                InfoText.Text = m_duration.ToString() + "/" + ChartArray.m_totalPoints.ToString();
                ResetPB.IsEnabled = true;
                RunningIndicator.IsRunning(false);
            }), DispatcherPriority.Background);
        }

      
        private void QuitPB_Click(object sender, RoutedEventArgs e)
        {          
            Close();
        }

        private void StartPB_Click(object sender, RoutedEventArgs e)
        {            
            m_sw = new Stopwatch();
            m_sw.Start();

            Task task = Task.Run(() => BackgroundTask(5000, 2));            
        }

        private void ResetPB_Click(object sender, RoutedEventArgs e)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                ChartArray.Reset();
                StartPB.IsEnabled = true;
                m_x = 0;
            }), DispatcherPriority.Background);

        }


        private void TestPB_Click(object sender, RoutedEventArgs e)
        {
            //Task.Run(() => TestRoutine());

            //visible = !visible;
            //ChartArray.SetTraceVisibility(WPFTools.MultiChartArray.SIGNAL_TYPE.RAW, 1, visible);

            //ChartArray.Refresh();
            //ChartArray.Resize();

            //int w = 0, h = 0;
            //ChartArray.GetBestBitmapSize(ref w, ref h);

            //ChartArray.Test();

            InfoText.Text = m_duration.ToString() + "/" + ChartArray.m_totalPoints.ToString();
        }


        
        private void TestRoutine()
        {
            int loopCount = 0;
            while (loopCount < 100)
            {
                Task task = Task.Run(() => BackgroundTask(1000, 5));
                task.Wait();

                ResetPB_Click(null, null);

                loopCount++;
            }
        }



        private UInt16[] SynthesizeGrayImage(int width, int height, UInt16 val)
        {
            UInt16[] image = new UInt16[width * height];

            for(int r = 0; r<height; r++)
                for(int c = 0; c<width; c++)
                {
                    int index = (r * width) + c;
                    image[index] = val;
                }

            return image;
        }

    }








    public class MainWindow_ViewModel : INotifyPropertyChanged
    {
        private int _width;
        public int width
        {
            get { return _width; }
            set { _width = value; OnPropertyChanged(new PropertyChangedEventArgs("width")); }
        }

        private int _height;
        public int height
        {
            get { return _height; }
            set { _height = value; OnPropertyChanged(new PropertyChangedEventArgs("height")); }
        }

        private WriteableBitmap _bitmap;
        public WriteableBitmap bitmap
        {
            get { return _bitmap; }
            set { _bitmap = value; OnPropertyChanged(new PropertyChangedEventArgs("bitmap")); }
        }

        public void SetBitmap(byte[] imageData, int newWidth, int newHeight)
        {
            if (newWidth != width || newHeight != height)
            {
                width = newWidth;
                height = newHeight;
                bitmap = BitmapFactory.New(width, height);
            }

            Int32Rect imageRect = new Int32Rect(0, 0, width, height);

            try
            {
                bitmap.Lock();
                bitmap.WritePixels(imageRect, imageData, width * 4, 0);
                bitmap.Unlock();
            }
            catch(Exception ex)
            {
                string errMsg = ex.Message;
            }
        }


        public byte[] SynthesizeImage(int width, int height)
        {
            byte[] data = new byte[width * height * 4];

            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    int ndx = (r * width * 4) + (c * 4);
                    data[ndx + 0] = 0;      // blue
                    data[ndx + 1] = 0;    // green
                    data[ndx + 2] = 0;      // red
                    data[ndx + 3] = 255;    // alpha
                }
            }

            return data;
        }


        public MainWindow_ViewModel()
        {
            width = 0;
            height = 0;

            int newWidth = 1200;
            int newHeight = 800;

            byte[] img = SynthesizeImage(newWidth, newHeight);

            SetBitmap(img, newWidth, newHeight);
        }


        public event PropertyChangedEventHandler PropertyChanged;
        public void OnPropertyChanged(PropertyChangedEventArgs e)
        {
            if (PropertyChanged != null)
            {
                PropertyChanged(this, e);
            }
        }
    }
}
