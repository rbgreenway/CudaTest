using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace WPFTools
{
    /// <summary>
    /// Interaction logic for AggregateChart.xaml
    /// </summary>
    public partial class AggregateChart : UserControl
    {
        public AggregateChart_ViewModel m_vm;

        public AggregateChart()
        {
            InitializeComponent();

            m_vm = new AggregateChart_ViewModel();
            DataContext = m_vm;
        }



        public void UpdateRanges(int xmax, int ymin, int ymax)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {             
                m_vm.xMaxText = xmax.ToString();
                m_vm.yMinText = ymin.ToString();
                m_vm.yMaxText = ymax.ToString();
            }), DispatcherPriority.Background);
        }

        private void AggregateImage_SizeChanged(object sender, SizeChangedEventArgs e)
        {

        }

        public void AddCheckBox(StackPanel checkboxStackpanel)
        {
            VisibilityStackPanel.Children.Add(checkboxStackpanel);
        }


        public StackPanel GetVisibilityStackPanel()
        {
            return VisibilityStackPanel;
        }


        public void SetHeaderText(string title)
        {
            AggregateHeaderText.Text = title;
        }

     
    }



    public class AggregateChart_ViewModel : INotifyPropertyChanged
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

        private string _yMaxText;
        public string yMaxText
        {
            get { return _yMaxText; }
            set { _yMaxText = value; OnPropertyChanged(new PropertyChangedEventArgs("yMaxText")); }
        }

        private string _yMinText;
        public string yMinText
        {
            get { return _yMinText; }
            set { _yMinText = value; OnPropertyChanged(new PropertyChangedEventArgs("yMinText")); }
        }

        private string _xMaxText;
        public string xMaxText
        {
            get { return _xMaxText; }
            set { _xMaxText = value; OnPropertyChanged(new PropertyChangedEventArgs("xMaxText")); }
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
            catch (Exception ex)
            {
                string errMsg = ex.Message;
            }
        }



        public AggregateChart_ViewModel()
        {
            width = 512;
            height = 512;
            bitmap = BitmapFactory.New(width, height);

            xMaxText = "XMax";
            yMinText = "YMin";
            yMaxText = "YMax";
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
