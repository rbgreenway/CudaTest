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
        public int m_width;      
        public int m_height;
        public string m_yMaxText;
        public string m_yMinText;
        public string m_xMaxText;
        public WriteableBitmap m_bitmap;
        


        public AggregateChart()
        {
            InitializeComponent();
            
            m_width = 512;
            m_height = 512;
            m_bitmap = BitmapFactory.New(m_width, m_height);

            SetRanges(1, 0, 1);
        }


        public void SetRanges(int xmax, int ymin, int ymax)
        {
            m_xMaxText = xmax.ToString();
            m_yMinText = ymin.ToString();
            m_yMaxText = ymax.ToString();

            XMaxText.Text = m_xMaxText;
            YMinText.Text = m_yMinText;
            YMaxText.Text = m_yMaxText;
        }


        public void UpdateRanges(int xmax, int ymin, int ymax)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                SetRanges(xmax, ymin, ymax);
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


        public void SetBitmap(byte[] imageData, int newWidth, int newHeight)
        {
            if (newWidth != m_width || newHeight != m_height)
            {
                m_width = newWidth;
                m_height = newHeight;
                m_bitmap = BitmapFactory.New(m_width, m_height);
                AggregateImage.Source = m_bitmap;
            }

            Int32Rect imageRect = new Int32Rect(0, 0, m_width, m_height);

            try
            {
                m_bitmap.Lock();
                m_bitmap.WritePixels(imageRect, imageData, m_width * 4, 0);
                m_bitmap.Unlock();
                AggregateImage.Source = m_bitmap;
            }
            catch (Exception ex)
            {
                string errMsg = ex.Message;
            }
        }

        public void SetBitmap(WriteableBitmap bitmap)
        {
            m_bitmap = bitmap;
            m_width = bitmap.PixelWidth;
            m_height = bitmap.PixelHeight;
            AggregateImage.Source = m_bitmap;
        }


    }

    


}
