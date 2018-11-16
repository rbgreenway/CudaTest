using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using CudaTools;
using System.ComponentModel;
using System.Windows.Threading;
using System.Threading.Tasks.Dataflow;
using System.Threading;
using System.IO;

namespace WPFTools
{
   
    public delegate void MultiChartArray_Message_EventHandler(object sender, MultiChartArray_Message_EventArgs e);
    public delegate void MultiChartArray_RangeChanged_EventHandler(object sender, MultiChartArray_RangeChanged_EventArgs e);

    public partial class MultiChartArray : UserControl
    {

        struct ButtonTag
        {
            public string type;
            public int position;
        };

        public class RangeClass
        {
            public int RangeMin { get; set; }
            public int RangeMax { get; set; }
        }

        public enum SIGNAL_TYPE
        {
            RAW = 0,
            STATIC_RATIO,
            CONTROL_SUBTRACTION,
            DYNAMIC_RATIO
        };

        public enum COMMAND_TYPE
        {
            RESIZE,
            REFRESH,
            RESET,
            SET_SELECTED
        }



        public event MultiChartArray_Message_EventHandler MessageEvent;
        protected virtual void OnMessage(MultiChartArray_Message_EventArgs e)
        {
            if (MessageEvent != null)
                MessageEvent(this, e);
        }

        public event MultiChartArray_RangeChanged_EventHandler RangeChangedEvent;
        protected virtual void OnRangeChanged(MultiChartArray_RangeChanged_EventArgs e)
        {
            if (RangeChangedEvent  != null)
                RangeChangedEvent(this, e);
        }




        private List<Button> m_columnButton;
        private List<Button> m_rowButton;

        int m_mouseDownRow, m_mouseUpRow;
        int m_mouseDownCol, m_mouseUpCol;
        Point m_mouseDownPoint;
        Point m_dragDown, m_drag;

        private bool[,] m_chartSelected;
        bool[] m_allChartsInRowSelected;
        bool[] m_allChartsInColumnSelected;
        bool m_allChartsSelected;

        int[] m_axisRange;

        System.Windows.Media.Color m_buttonColorSelected;
        System.Windows.Media.Color m_buttonColorNotSelected;

        private SIGNAL_TYPE m_visibleSignal;

        public DispatcherTimer m_refreshTimer;

        public int m_numPoints;

        Dictionary<SIGNAL_TYPE, ChartArray> m_chartArrays;

        TaskScheduler m_uiTask;

        CancellationTokenSource m_tokenSource;

        public ITargetBlock<Tuple<int[], int[], SIGNAL_TYPE, int>> m_dataPipeline;
        public ITargetBlock<Tuple<int[], int[], SIGNAL_TYPE, int, COMMAND_TYPE>> m_guiPipeline;
        bool m_newDataAdded;

        public int m_totalPoints;

        ImageTool m_imageTool;


        /// <summary>
        /// /////////////////////////////////////////////////////////////////////////////////////////////////
        /// </summary>


        public AggregateChart m_aggregateChart;
        public int m_rows;
        public int m_cols;
        public int m_numTraces;
        public Dictionary<int, MultiChartArray_TraceItem> m_traces;
        public Dictionary<Tuple<int, MultiChartArray.SIGNAL_TYPE>, Color> m_traceColors;
        public int m_maxPoints;
        public int m_chartArrayWidth;
        public int m_chartArrayHeight;
        public int m_padding;
        public int m_margin;
        public WriteableBitmap m_bitmap;
        public WriteableBitmap m_overlay;

        ///////////////////////////////////////////////////////////////////////////////////////////////////////


        public MultiChartArray()
        {
            InitializeComponent();

            m_axisRange = new int[4];

            m_uiTask = TaskScheduler.FromCurrentSynchronizationContext();

            m_tokenSource = new CancellationTokenSource();

            m_imageTool = new ImageTool();
        }


        public void Init(int rows, int cols, int maxNumPoints, List<MultiChartArray_TraceItem> traces, AggregateChart aggregateChart)
        {
            int numTraces = traces.Count;
            int margin = 1;
            int padding = 3;
            
            ////////////////////////////////////////////////////////////////////////////////////////////////


            m_chartArrayWidth = 0;
            m_chartArrayHeight = 0;

            m_cols = cols;
            m_rows = rows;
            m_padding = padding;
            m_margin = margin;
            m_maxPoints = maxNumPoints;
            m_numTraces = traces.Count;
            m_aggregateChart = aggregateChart;

            // initialize traces and trace colors
            int traceNum = 0;
            m_traces = new Dictionary<int, MultiChartArray_TraceItem>();
            m_traceColors = new Dictionary<Tuple<int, MultiChartArray.SIGNAL_TYPE>, Color>();
            foreach (MultiChartArray_TraceItem item in traces)
            {
                item.traceNum = traceNum;
                m_traces.Add(item.id, item);
                traceNum++;

                // initialize trace colors                    
                foreach (int value in Enum.GetValues(typeof(MultiChartArray.SIGNAL_TYPE)))
                {
                    MultiChartArray.SIGNAL_TYPE signal = (MultiChartArray.SIGNAL_TYPE)value;
                    m_traceColors.Add(Tuple.Create<int, MultiChartArray.SIGNAL_TYPE>(item.id, signal), Colors.Yellow);  // default to yellow
                }
            }



            int pixelsPerChart = 42;

            int newWidth =  (pixelsPerChart * m_cols) + ((m_cols - 1) * m_padding) + (2 * m_margin);
            int newHeight = (pixelsPerChart * m_rows) + ((m_rows - 1) * m_padding) + (2 * m_margin);

            byte[] img = SynthesizeImage(newWidth, newHeight);

            SetBitmap(img, newWidth, newHeight);

            if (aggregateChart != null)
            {
                aggregateChart.SetBitmap(img, newWidth / 2, newHeight / 2);
                aggregateChart.SetRanges(1, 0, 1);
            }
            

            ////////////////////////////////////////////////////////////////////////////////////////

            m_mouseDownRow = -1;
            m_mouseDownCol = -1;

            m_numPoints = 0;

            InitializeComponent();

            m_visibleSignal = SIGNAL_TYPE.RAW;

            m_buttonColorNotSelected = Colors.LightGray;
            m_buttonColorSelected = Colors.Red;

            m_rowButton = new List<Button>();
            m_columnButton = new List<Button>();


            BuildChartArray();
            SetupVisiblityCheckBoxes(m_traces);

            m_refreshTimer = new DispatcherTimer();
            m_refreshTimer.Tick += M_refreshTimer_Tick;
            m_refreshTimer.Interval = TimeSpan.FromMilliseconds(100);
            m_refreshTimer.Start();

            m_newDataAdded = false;
            m_dataPipeline = CreateDataPipeline(m_tokenSource.Token, m_chartArrays);
            m_guiPipeline = CreateGuiPipeline(m_uiTask, m_tokenSource.Token, m_chartArrays, aggregateChart);

            if(aggregateChart != null)
            {
                aggregateChart.SizeChanged += AggregateChart_SizeChanged;
            }




        }

     

        public void AppendData(int[] x, int[] y, SIGNAL_TYPE signal, int id)
        {
            // add data to chart array (the x array is superfluous since it contains all the same values, should just be an int, not int[])
            //m_chartArrays[signal].AppendData(x, y);
            if (m_traces.ContainsKey(id))
            {
                m_dataPipeline.Post(Tuple.Create<int[], int[], SIGNAL_TYPE, int>(x, y, signal, m_traces[id].traceNum));
                m_newDataAdded = true;
            }
        }


        public void Refresh()
        {  
            if(m_guiPipeline != null)         
                m_guiPipeline.Post(Tuple.Create<int[], int[], SIGNAL_TYPE, int, COMMAND_TYPE>(null, null, m_visibleSignal, 0, COMMAND_TYPE.REFRESH));
        }



        private BitmapImage ConvertWriteableBitmapToBitmapImage(WriteableBitmap wbm)
        {
            BitmapImage bmImage = new BitmapImage();
            using (MemoryStream stream = new MemoryStream())
            {
                PngBitmapEncoder encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(wbm));
                encoder.Save(stream);
                bmImage.BeginInit();
                bmImage.CacheOption = BitmapCacheOption.OnLoad;
                bmImage.StreamSource = stream;
                bmImage.EndInit();
                bmImage.Freeze();
            }
            return bmImage;
        }

        private void PrintBitmapSource(BitmapSource inBms)
        {           
            var pd = new PrintDialog();
            var ret = pd.ShowDialog();
            if (ret.Value)
            {
                var dv = new DrawingVisual();
                using (var dc = dv.RenderOpen())
                {
                    dc.DrawImage(inBms, new Rect(0, 0, inBms.Width, inBms.Height));
                }
                pd.PrintVisual(dv, "document image");
            }
        }



        public void UpdateAggregateRange()
        {
            if(m_aggregateChart != null)
            {
                m_chartArrays[m_visibleSignal].GetRanges(ref m_axisRange);
                m_aggregateChart.UpdateRanges(m_axisRange[1], m_axisRange[2], m_axisRange[3]);
            }
        }

        public int NumRows()
        {
            return m_rows;
        }

        public int NumCols()
        {
            return m_cols;
        }

        public int MaxNumPoints()
        {
            return m_maxPoints;
        }


        // //////////////////////////////////////////////////////////////////////////////////////
        // //////////////////////////////////////////////////////////////////////////////////////

        public void BuildChartArray()
        {
            m_chartArrays = new Dictionary<SIGNAL_TYPE, ChartArray>();
            int initialYmax = (m_maxPoints > 100) ? 100 : m_maxPoints;
            int initialXmax = 10;
            int aggregateWidth = 256;
            int aggregateHeight = 256;
            if(m_aggregateChart != null) { aggregateWidth = m_aggregateChart.m_width; aggregateHeight = m_aggregateChart.m_height; }

            foreach (int value in Enum.GetValues(typeof(SIGNAL_TYPE)))
            {
                SIGNAL_TYPE signal = (SIGNAL_TYPE)value;
                m_chartArrays.Add(signal, new ChartArray());
                m_chartArrays[signal].Init(m_rows, m_cols, m_chartArrayWidth, m_chartArrayHeight, m_margin, m_padding,
                       aggregateWidth, aggregateHeight,
                       Colors.DarkBlue, Colors.Black, Color.FromArgb(255, 85, 85, 85), Colors.Black, Colors.White,
                       Colors.Yellow,
                       0, initialXmax, 0, initialYmax, m_maxPoints, m_numTraces);
                m_chartArrays[signal].Redraw();
                m_chartArrays[signal].RedrawAggregate();
            }

            int maxTraces = m_chartArrays[SIGNAL_TYPE.RAW].GetMaxNumberOfTraces();
            if (m_numTraces > maxTraces) m_numTraces = maxTraces;

            WriteableBitmap bmap = m_bitmap;
            m_chartArrays[m_visibleSignal].Refresh(ref bmap);


            if (m_aggregateChart != null)
            {
                WriteableBitmap aggregateBitmapRef = m_aggregateChart.m_bitmap;
                m_chartArrays[m_visibleSignal].RefreshAggregate(ref aggregateBitmapRef);
            }

            // creat the range of each series
            m_chartSelected = new bool[m_rows, m_cols];

            m_allChartsInColumnSelected = new bool[m_cols];
            m_allChartsInRowSelected = new bool[m_rows];
            SetUpChartArrayButtons();
        }


        public void Reset()
        {
            m_guiPipeline.Post(Tuple.Create<int[], int[], SIGNAL_TYPE, int, COMMAND_TYPE>(null, null, m_visibleSignal, 0, COMMAND_TYPE.RESET));    
        }



        public void SetDefaultChartRanges(SIGNAL_TYPE signal, int xmin, int xmax, int ymin, int ymax)
        {
            m_chartArrays[signal].SetIntialRanges(xmin, xmax, ymin, ymax);
            m_guiPipeline.Post(Tuple.Create<int[], int[], SIGNAL_TYPE, int, COMMAND_TYPE>(null, null, signal, 0, COMMAND_TYPE.RESET));
        }


        public void SetTraceColor(SIGNAL_TYPE signal, int traceNum, Color color)
        {
            m_chartArrays[signal].SetTraceColor(traceNum, color);

            // find trace id from the given signal and traceNum
            foreach(KeyValuePair<int,MultiChartArray_TraceItem> item in m_traces)
            {
                if(item.Value.traceNum == traceNum)
                {
                    int id = item.Value.id;
                    m_traceColors[Tuple.Create<int, SIGNAL_TYPE>(id, signal)] = color;
                    break;
                }
            }

            UpdateTraceColorLegend(); 
        }


        public void SetTraceVisibility(SIGNAL_TYPE signal, int traceNum, bool isVisible)
        {
            m_chartArrays[signal].SetTraceVisibility(traceNum, isVisible);
        }

        private void SetupVisiblityCheckBoxes(Dictionary<int, MultiChartArray_TraceItem> traces)
        {
            foreach(KeyValuePair<int,MultiChartArray_TraceItem> item in traces)
            {
                StackPanel sp = new StackPanel();
                sp.Margin = new Thickness(10, 2, 10, 2);
                sp.Tag = item.Value.id;
                sp.Orientation = Orientation.Horizontal;

                CheckBox cb = new CheckBox();
                cb.Content = item.Value.name;
                cb.Tag = item.Value.id;
                cb.Margin = new Thickness(0, 2, 0, 2);
                cb.Checked += VisibilityCB_CheckChanged;
                cb.Unchecked += VisibilityCB_CheckChanged;
                cb.IsChecked = true;
                sp.Children.Add(cb);

                Rectangle rect = new Rectangle();
                rect.Tag = item.Value.id;
                rect.Width = 10;
                rect.Height = 4;
                rect.Fill = new SolidColorBrush(Colors.Yellow);
                rect.Margin = new Thickness(4, 0, 0, 0);                
                sp.Children.Add(rect);

                if (m_aggregateChart != null)
                {
                    m_aggregateChart.AddCheckBox(sp);                    
                }
                else
                {
                    // TODO:  add place to put visibility checkboxes  if there is not an aggregate chart
                }
            }
        }




        private void UpdateTraceColorLegend()
        {
            SIGNAL_TYPE signal = m_visibleSignal;


            if(m_aggregateChart != null)
            {
                StackPanel VisibilityStackPanel = m_aggregateChart.GetVisibilityStackPanel();

                foreach (KeyValuePair<int, MultiChartArray_TraceItem> item in m_traces)
                {
                    int id = item.Value.id;
                    Color color = m_traceColors[Tuple.Create<int, SIGNAL_TYPE>(id, signal)];

                    // find the visibility checkbox stackpanel with this id
                    foreach (UIElement child1 in VisibilityStackPanel.Children)
                    {
                        if (child1.GetType() == typeof(StackPanel))
                        {
                            StackPanel stackpanel = (StackPanel)child1;
                            foreach (UIElement child2 in stackpanel.Children)
                            {
                                if (child2.GetType() == typeof(Rectangle))
                                {
                                    Rectangle rect = (Rectangle)child2;
                                    if ((int)rect.Tag == id)
                                    {
                                        rect.Fill = new SolidColorBrush(color);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                // TODO: add place to put visibility checkboxes  if there is not an aggregate chart
            }

            
        }



        private void VisibilityCB_CheckChanged(object sender, RoutedEventArgs e)
        {
            if(e.Source.GetType() == typeof(CheckBox))
            {
                CheckBox cb = (CheckBox)e.Source;
                int id = (int)cb.Tag;

                foreach(KeyValuePair<int, MultiChartArray_TraceItem> item in m_traces)
                {
                    if(item.Value.id == id)
                    {
                        SetTraceVisibility(SIGNAL_TYPE.RAW, item.Value.traceNum, (bool)cb.IsChecked);
                        SetTraceVisibility(SIGNAL_TYPE.CONTROL_SUBTRACTION, item.Value.traceNum, (bool)cb.IsChecked);
                        SetTraceVisibility(SIGNAL_TYPE.STATIC_RATIO, item.Value.traceNum, (bool)cb.IsChecked);
                        SetTraceVisibility(SIGNAL_TYPE.DYNAMIC_RATIO, item.Value.traceNum, (bool)cb.IsChecked);
                        Refresh();
                        break;
                    }
                }
            }
        }


        private void SetUpChartArrayButtons()
        {
            // clear any previously set up buttons from grids
            ColumnButtonGrid.Children.Clear();
            ColumnButtonGrid.ColumnDefinitions.Clear();
            RowButtonGrid.Children.Clear();
            RowButtonGrid.RowDefinitions.Clear();
            m_rowButton.Clear();
            m_columnButton.Clear();

            SolidColorBrush brush = new SolidColorBrush(m_buttonColorNotSelected);

            double fontSize;
            if (m_cols < 32)
                fontSize = 10;
            else
                fontSize = 6;
            

            for (int i = 0; i < m_cols; i++)
            {
                ColumnDefinition colDef = new ColumnDefinition();
                ColumnButtonGrid.ColumnDefinitions.Add(colDef);
                Button button = new Button();
                ButtonTag tag = new ButtonTag();
                tag.type = "C";
                tag.position = i;
                button.Tag = tag;
               
                TextBlock tb = new TextBlock();
                tb.FontSize = fontSize;                            
                tb.Text = (i + 1).ToString();
                button.Content = tb;
               
                button.Click += button_Click;
                ColumnButtonGrid.Children.Add(button);
                Grid.SetColumn(button, i);
                Grid.SetRow(button, 0);
                m_columnButton.Add(button);
                button.Background = brush;
            }

            for (int i = 0; i < m_rows; i++)
            {
                RowDefinition rowDef = new RowDefinition();
                RowButtonGrid.RowDefinitions.Add(rowDef);
                Button button = new Button();
                ButtonTag tag = new ButtonTag();
                tag.type = "R";
                tag.position = i;
                button.Tag = tag;
                int unicode;
                char character;
                string text;
                if (i < 26)
                {
                    unicode = 65 + i;
                    character = (char)unicode;
                    text = character.ToString();
                }
                else
                {
                    unicode = 39 + i;
                    character = (char)unicode;
                    text = "A" + character.ToString();
                }

                TextBlock tb = new TextBlock();
                tb.FontSize = fontSize;
                tb.Text = text;
                button.Content = tb;

                button.Click += button_Click;

                RowButtonGrid.Children.Add(button);
                Grid.SetColumn(button, 0);
                Grid.SetRow(button, i);
                m_rowButton.Add(button);
                button.Background = brush;
            }
        }



        bool IsRowSeleted(int row)
        {
            bool result = true;

            for (int c = 0; c < m_cols; c++)
            {
                if (!m_chartSelected[row, c])
                {
                    result = false;
                    break;
                }
            }

            return result;
        }


        bool IsColumnSeleted(int col)
        {
            bool result = true;

            for (int r = 0; r < m_rows; r++)
            {
                if (!m_chartSelected[r, col])
                {
                    result = false;
                    break;
                }
            }

            return result;
        }


        bool AreAllSelected()
        {
            bool result = true;

            for (int r = 0; r < m_rows; r++)
            {
                if (!IsRowSeleted(r))
                {
                    result = false;
                    break;
                }
            }

            return result;
        }


        void SetButtonStates()
        {
            SolidColorBrush brushSelected = new SolidColorBrush(m_buttonColorSelected);
            SolidColorBrush brushNotSelected = new SolidColorBrush(m_buttonColorNotSelected);
            bool all = true;
            for (int r = 0; r < m_rows; r++)
            {
                m_allChartsInRowSelected[r] = IsRowSeleted(r);
                if (!m_allChartsInRowSelected[r])
                {
                    all = false;
                    m_rowButton.ElementAt(r).Background = brushNotSelected;
                }
                else m_rowButton.ElementAt(r).Background = brushSelected;
            }

            for (int c = 0; c < m_cols; c++)
            {
                m_allChartsInColumnSelected[c] = IsColumnSeleted(c);
                if (!m_allChartsInColumnSelected[c])
                {
                    all = false;
                    m_columnButton.ElementAt(c).Background = brushNotSelected;
                }
                else m_columnButton.ElementAt(c).Background = brushSelected;
            }

            m_allChartsSelected = all;

            if (m_allChartsSelected)
                SelectAllPB.Background = brushSelected;
            else
                SelectAllPB.Background = brushNotSelected;
        }



        private void SelectAllPB_Click(object sender, RoutedEventArgs e)
        {

            bool state;
            if (AreAllSelected()) state = false; else state = true;

            for (int c = 0; c < m_cols; c++)
            {
                for (int r = 0; r < m_rows; r++)
                {
                    m_chartSelected[r, c] = state;
                }
            }

            SetButtonStates();
            UpdateSelectedCharts();
        }



        // this function gets called whenever a row or column button is clicked
        void button_Click(object sender, RoutedEventArgs e)
        {
            Button button = (Button)sender;
            ButtonTag tag = (ButtonTag)(((Button)sender).Tag);
            bool state;

            if (tag.type == "C")  // it's a column button
            {
                int columnNumber = tag.position;
                if (m_allChartsInColumnSelected[columnNumber]) state = false; else state = true;

                for (int r = 0; r < m_rows; r++)
                {
                    m_chartSelected[r, columnNumber] = state;
                }
            }
            else if (tag.type == "R") // it's a row button
            {
                int rowNumber = tag.position;
                if (m_allChartsInRowSelected[rowNumber]) state = false; else state = true;

                for (int c = 0; c < m_cols; c++)
                {
                    m_chartSelected[rowNumber, c] = state;
                }
            }

            UpdateSelectedCharts();

            SetButtonStates();

        }



        // //////////////////////////////////////////////////////////////////////////////////////
        // //////////////////////////////////////////////////////////////////////////////////////





        private void gridChart_SizeChanged(object sender, SizeChangedEventArgs e)
        {

        }



        void ChartArray_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            // find m_mouseUpCol and m_mouseUpRow

            if (m_mouseDownRow != -1 && m_mouseDownCol != -1) // make sure we have a captured mouse down first
            {
                m_mouseDownPoint = e.GetPosition(overlayBitmap);

                int x = (int)(m_mouseDownPoint.X / overlayBitmap.ActualWidth * m_overlay.PixelWidth);
                int y = (int)(m_mouseDownPoint.Y / overlayBitmap.ActualHeight * m_overlay.PixelHeight);

                m_mouseUpRow = m_chartArrays[m_visibleSignal].GetRowFromY(y);
                m_mouseUpCol = m_chartArrays[m_visibleSignal].GetColumnFromX(x);

                if (m_mouseUpRow != -1)
                {
                    // set the Band(s) for the selected charts
                    int rowStart, rowStop;
                    int colStart, colStop;

                    if (m_mouseUpCol < m_mouseDownCol)
                    {
                        colStart = m_mouseUpCol; colStop = m_mouseDownCol;
                    }
                    else
                    {
                        colStart = m_mouseDownCol; colStop = m_mouseUpCol;
                    }

                    if (m_mouseUpRow < m_mouseDownRow)
                    {
                        rowStart = m_mouseUpRow; rowStop = m_mouseDownRow;
                    }
                    else
                    {
                        rowStart = m_mouseDownRow; rowStop = m_mouseUpRow;
                    }

                    SelectCharts(rowStart, rowStop, colStart, colStop);

                }

                // reset the mouse down locations
                m_mouseDownRow = -1;
                m_mouseDownCol = -1;

                SetButtonStates();

            }

            m_overlay.Clear();

            e.Handled = true;

        }



        void ChartArray_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            m_mouseDownPoint = e.GetPosition(overlayBitmap);

            int x = (int)(m_mouseDownPoint.X / overlayBitmap.ActualWidth * m_overlay.PixelWidth);
            int y = (int)(m_mouseDownPoint.Y / overlayBitmap.ActualHeight * m_overlay.PixelHeight);

            m_mouseDownCol = m_chartArrays[m_visibleSignal].GetColumnFromX(x);
            m_mouseDownRow = m_chartArrays[m_visibleSignal].GetRowFromY(y);

            m_dragDown.X = (double)x;
            m_dragDown.Y = (double)y;

            e.Handled = true;

        }



        private void ChartArrayGrid_MouseLeave(object sender, MouseEventArgs e)
        {
            m_overlay.Clear();

            // reset the mouse down locations
            m_mouseDownRow = -1;
            m_mouseDownCol = -1;
        }



        private void ChartArrayGrid_MouseMove(object sender, MouseEventArgs e)
        {
            m_overlay.Clear();

            if (m_mouseDownRow != -1) // mouse button is down
            {
                Point pt = e.GetPosition(overlayBitmap);

                m_drag.X = pt.X / overlayBitmap.ActualWidth * m_overlay.PixelWidth;
                m_drag.Y = pt.Y / overlayBitmap.ActualHeight * m_overlay.PixelHeight;

                int x1, x2, y1, y2;

                if (m_dragDown.X < m_drag.X)
                {
                    x1 = (int)(m_dragDown.X);
                    x2 = (int)(m_drag.X);
                }
                else
                {
                    x1 = (int)(m_drag.X);
                    x2 = (int)(m_dragDown.X);
                }


                if (m_dragDown.Y < m_drag.Y)
                {
                    y1 = (int)(m_dragDown.Y);
                    y2 = (int)(m_drag.Y);
                }
                else
                {
                    y1 = (int)(m_drag.Y);
                    y2 = (int)(m_dragDown.Y);
                }

                m_overlay.DrawRectangle(x1, y1, x2, y2, Colors.Red);
                m_overlay.DrawRectangle(x1 + 1, y1 + 1, x2 - 1, y2 - 1, Colors.Red);
            }
        }


        public void SelectCharts(int startRow, int endRow, int startCol, int endCol)
        {
            if (startRow != -1 && endRow != -1 && startCol != -1 && endCol != -1)
            {
                for (int r = startRow; r < endRow + 1; r++)
                    for (int c = startCol; c < endCol + 1; c++)
                    {
                        m_chartSelected[r, c] = !m_chartSelected[r, c];
                    }

                UpdateSelectedCharts();
            }
        }


        private void AggregateChart_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            if (m_aggregateChart != null)
            {
                int w = (int)imageBitmap.ActualWidth;
                int h = (int)imageBitmap.ActualHeight;
                int aggw = (int)m_aggregateChart.AggregateImage.ActualWidth;
                int aggh = (int)m_aggregateChart.AggregateImage.ActualHeight;
                if (w == 0)
                {
                    w = 512; h = 512;
                    aggw = 256; aggh = 256;
                }


                // post a resize action to the chart data pipeline
                int[] x = new int[2] { w * 2, aggw * 2 };
                int[] y = new int[2] { h * 2, aggh * 2 };
                m_guiPipeline.Post(Tuple.Create<int[], int[], SIGNAL_TYPE, int, COMMAND_TYPE>(x, y, SIGNAL_TYPE.RAW, 0, COMMAND_TYPE.RESIZE));
            }
        }

   
        private void M_refreshTimer_Tick(object sender, EventArgs e)
        {
            if (m_newDataAdded)
            {
                m_guiPipeline.Post(Tuple.Create<int[], int[], SIGNAL_TYPE, int, COMMAND_TYPE>(null, null, m_visibleSignal, 0, COMMAND_TYPE.REFRESH));
                m_newDataAdded = false;
            }
        }

        public void GetBestBitmapSize(ref int width, ref int height, ref int chartPanelWidth, ref int chartPanelHeight)
        {
            int w = 0, h = 0, pw = 0, ph = 0;
            OptimizeBitmapSize((int)imageBitmap.ActualWidth, (int)imageBitmap.ActualHeight, ref w, ref h, ref pw, ref ph);
            width = w;
            height = h;
            chartPanelWidth = pw;
            chartPanelHeight = ph;
        }


        public void UpdateSelectedCharts()
        {
            m_guiPipeline.Post(Tuple.Create<int[], int[], SIGNAL_TYPE, int, COMMAND_TYPE>(null, null, m_visibleSignal, 0, COMMAND_TYPE.SET_SELECTED));
        }



        private void AnalysisRadioButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender.GetType() != typeof(RadioButton)) return;

            RadioButton rb = (RadioButton)sender;
            string tag = (string)rb.Tag;

            switch (tag)
            {
                case "Raw":
                    m_visibleSignal = SIGNAL_TYPE.RAW;
                    m_aggregateChart.SetHeaderText("Raw");
                    break;
                case "StaticRatio":
                    m_visibleSignal = SIGNAL_TYPE.STATIC_RATIO;
                    m_aggregateChart.SetHeaderText("Static Ratio");                    
                    break;
                case "ControlSubtraction":
                    m_visibleSignal = SIGNAL_TYPE.CONTROL_SUBTRACTION;
                    m_aggregateChart.SetHeaderText("Control Subtraction");                    
                    break;
                case "DynamicRatio":
                    m_visibleSignal = SIGNAL_TYPE.DYNAMIC_RATIO;
                    m_aggregateChart.SetHeaderText("Dynamic Ratio");
                    break;
                default:
                    m_visibleSignal = SIGNAL_TYPE.RAW;
                    m_aggregateChart.SetHeaderText("Raw");
                    break;
            }
   
            m_guiPipeline.Post(Tuple.Create<int[], int[], SIGNAL_TYPE, int, COMMAND_TYPE>(null, null, m_visibleSignal, 0, COMMAND_TYPE.REFRESH));

            UpdateTraceColorLegend();

        }






        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        public ITargetBlock<Tuple<int[], int[], SIGNAL_TYPE, int>> CreateDataPipeline(CancellationToken cancelToken, 
                                Dictionary<SIGNAL_TYPE, ChartArray> _charts)
        {
            Dictionary<SIGNAL_TYPE, ChartArray> charts = _charts;
            int[] axisRange = new int[4];        


            var AddData = new ActionBlock<Tuple<int[], int[], SIGNAL_TYPE, int>>(inputData =>
            {
                // INPUTS:
                //  item 1 - x data array
                //  item 2 - y data array
                //  item 3 - the data's signal type, i.e. RAW, STATIC_RATIO, CONTROL_SUBTRACTION, or DYNAMIC_RATIO
                //  item 4 - the index of the indicator to which this data belongs
            

                int[] x = inputData.Item1;
                int[] y = inputData.Item2;
                SIGNAL_TYPE signalType = inputData.Item3;
                int indicatorNdx = inputData.Item4;

                try
                {
                    if(signalType == SIGNAL_TYPE.RAW) m_totalPoints++;
                    charts[signalType].AppendData(x, y, indicatorNdx);                   
                }
                catch (OperationCanceledException)
                {
                    return;
                }
            },           
            new ExecutionDataflowBlockOptions
            {
               // TaskScheduler = uiTask,
                CancellationToken = cancelToken,                
                MaxDegreeOfParallelism = 1
            });

            
            return AddData;
        }






        public ITargetBlock<Tuple<int[], int[], SIGNAL_TYPE, int, COMMAND_TYPE>> CreateGuiPipeline(TaskScheduler uiTask,
                           CancellationToken cancelToken,
                           Dictionary<SIGNAL_TYPE, ChartArray> _charts,
                           AggregateChart _aggregateChart)
        {
            Dictionary<SIGNAL_TYPE, ChartArray> charts = _charts;
            AggregateChart aggregateChart = _aggregateChart;
            SIGNAL_TYPE visibleSignal = SIGNAL_TYPE.RAW;
            int[] axisRange = new int[4];

            var GuiUpdates = new ActionBlock<Tuple<int[], int[], SIGNAL_TYPE, int, COMMAND_TYPE>>(inputData =>
            {
                // INPUTS:
                //  item 1 - x data array
                //  item 2 - y data array
                //  item 3 - the data's signal type, i.e. RAW, STATIC_RATIO, CONTROL_SUBTRACTION, or DYNAMIC_RATIO
                //  item 4 - the index of the indicator to which this data belongs
                //  item 5 - command type that is put on the queue. Depending on what this is, the previous parameters are
                //           interpreted differently.  For example, if it is RESIZE, the new bitmap dimensions should be 
                //           in x and y (items 1 and 2 above).  The new chartArray should be x[0], y[0], 
                //           and the new aggregate chart should be x[1], y[1].

                if (inputData == null) return;

                int[] x = inputData.Item1;
                int[] y = inputData.Item2;
                SIGNAL_TYPE signalType = inputData.Item3;
                int indicatorNdx = inputData.Item4;
                COMMAND_TYPE commandType = inputData.Item5;

                try
                {
                    switch (commandType)
                    {
                        case COMMAND_TYPE.RESIZE:
                            // resize chartArray and aggregate bitmaps                                              
                            // find the optimal size to best fit the Actual window size

                            int pixelWidthPerChart = (x[0] - (2 * m_margin) - ((m_cols - 1) * m_padding)) / m_cols;
                            int pixelHeightPerChart = (y[0] - (2 * m_margin) - ((m_rows - 1) * m_padding)) / m_rows;

                            int w = (pixelWidthPerChart * m_cols) + ((m_cols - 1) * m_padding) + (2 * m_margin);
                            int h = (pixelHeightPerChart * m_rows) + ((m_rows - 1) * m_padding) + (2 * m_margin);

                            if (w != m_chartArrayWidth || h != m_chartArrayHeight)
                            {
                                m_chartArrayWidth = w;
                                m_chartArrayHeight = h;
                                if (aggregateChart != null)
                                {
                                    aggregateChart.SetBitmap(BitmapFactory.New(x[1], y[1]));                                   
                                }

                                m_bitmap = BitmapFactory.New(w, h);
                                m_overlay = BitmapFactory.New(w, h);
                                imageBitmap.Source = m_bitmap;
                                overlayBitmap.Source = m_overlay;
                               

                                foreach (int value in Enum.GetValues(typeof(SIGNAL_TYPE)))
                                {
                                    SIGNAL_TYPE signal = (SIGNAL_TYPE)value;
                                    charts[signal].Resize(w, h, x[1], y[1]);
                                }
                            }

                            // refresh chart array
                            charts[visibleSignal].Redraw();
                            WriteableBitmap bitmapRef4 = m_bitmap;
                            charts[visibleSignal].Refresh(ref bitmapRef4);

                            // refresh aggregate chart
                            if (aggregateChart != null)
                            {
                                charts[visibleSignal].RedrawAggregate();
                                WriteableBitmap aggregateBitmapRef4 = aggregateChart.m_bitmap;
                                charts[visibleSignal].RefreshAggregate(ref aggregateBitmapRef4);
                            }
                            
                            break;

                        case COMMAND_TYPE.REFRESH:
                            visibleSignal = signalType;

                            // refresh chart array image
                            charts[visibleSignal].Redraw();
                            WriteableBitmap bitmapRef1 = m_bitmap;
                            charts[visibleSignal].Refresh(ref bitmapRef1);

                            // refresh aggregate image
                            if (aggregateChart != null)
                            {
                                charts[visibleSignal].RedrawAggregate();
                                WriteableBitmap aggregateBitmapRef4 = aggregateChart.m_bitmap;
                                charts[visibleSignal].RefreshAggregate(ref aggregateBitmapRef4);

                                // update the range labels                               
                                charts[visibleSignal].GetRanges(ref axisRange);
                                aggregateChart.SetRanges(axisRange[1], axisRange[2], axisRange[3]);
                            }
                            
                            break;

                        case COMMAND_TYPE.RESET:
                            // clears data from all charts   

                            foreach (int value in Enum.GetValues(typeof(SIGNAL_TYPE)))
                            {
                                SIGNAL_TYPE signal = (SIGNAL_TYPE)value;
                                charts[signal].Reset();

                                charts[signal].Redraw();
                                charts[signal].RedrawAggregate();
                            }
                          

                            WriteableBitmap bitmapRef2 = m_bitmap;
                            charts[visibleSignal].Refresh(ref bitmapRef2);

                            if (aggregateChart != null)
                            {
                                WriteableBitmap aggregateBitmapRef2 = aggregateChart.m_bitmap;
                                charts[visibleSignal].RefreshAggregate(ref aggregateBitmapRef2);

                                // update the range labels                               
                                charts[visibleSignal].GetRanges(ref axisRange);
                                aggregateChart.SetRanges(axisRange[1], axisRange[2], axisRange[3]);
                            }

                            m_totalPoints = 0;
                            break;

                        case COMMAND_TYPE.SET_SELECTED:
                            // have to convert to a 1D bool array, because that's what is needed by the C++ function
                            bool[] temp = new bool[m_rows * m_cols];
                            for (int r = 0; r < m_rows; r++)
                                for (int c = 0; c < m_cols; c++)
                                {
                                    temp[r * m_cols + c] = m_chartSelected[r, c];
                                }

                            foreach (int value in Enum.GetValues(typeof(SIGNAL_TYPE)))
                            {
                                SIGNAL_TYPE signal = (SIGNAL_TYPE)value;

                                // refresh chart array
                                charts[signal].Redraw();

                                if (signal == visibleSignal)
                                {
                                    WriteableBitmap bitmapRef3 = m_bitmap;
                                    charts[signal].SetSelectedCharts(temp, ref bitmapRef3);
                                }
                                else
                                {
                                    charts[signal].SetSelectedCharts(temp);
                                }


                                if (aggregateChart != null)
                                {
                                    // refresh aggregate chart
                                    charts[signal].RedrawAggregate();

                                    if (signal == visibleSignal)
                                    {
                                        WriteableBitmap aggregateBitmapRef3 = aggregateChart.m_bitmap;
                                        charts[visibleSignal].RefreshAggregate(ref aggregateBitmapRef3);
                                    }
                                }
                                
                            }

                            

                            break;
                    }
                }
                catch (OperationCanceledException)
                {
                    return;
                }
            },
            new ExecutionDataflowBlockOptions
            {
                TaskScheduler = uiTask,
                CancellationToken = cancelToken,
                MaxDegreeOfParallelism = 4
            });


            return GuiUpdates;
        }




        public void SetBitmap(byte[] imageData, int newWidth, int newHeight)
        {
            if (newWidth != m_chartArrayWidth || newHeight != m_chartArrayHeight)
            {
                m_chartArrayWidth = newWidth;
                m_chartArrayHeight = newHeight;
                m_bitmap = BitmapFactory.New(m_chartArrayWidth, m_chartArrayHeight);
                imageBitmap.Source = m_bitmap;
                m_overlay = BitmapFactory.New(m_chartArrayWidth, m_chartArrayHeight);
                overlayBitmap.Source = m_overlay;
            }

            Int32Rect imageRect = new Int32Rect(0, 0, m_chartArrayWidth, m_chartArrayHeight);

            try
            {
                m_bitmap.Lock();
                m_bitmap.WritePixels(imageRect, imageData, m_chartArrayWidth * 4, 0);
                m_bitmap.Unlock();
                imageBitmap.Source = m_bitmap;
            }
            catch (Exception ex)
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

        public void OptimizeBitmapSize(int windowWidth, int windowHeight,
            ref int suggestedBitmapWidth, ref int suggestedBitmapHeight,
            ref int pixelWidthPerChart, ref int pixelHeightPerChart)
        {
            pixelWidthPerChart = (windowWidth - (2 * m_margin) - ((m_cols - 1) * m_padding)) / m_cols;
            pixelHeightPerChart = (windowHeight - (2 * m_margin) - ((m_rows - 1) * m_padding)) / m_rows;

            suggestedBitmapWidth = (pixelWidthPerChart * m_cols) + ((m_cols - 1) * m_padding) + (2 * m_margin);
            suggestedBitmapHeight = (pixelHeightPerChart * m_rows) + ((m_rows - 1) * m_padding) + (2 * m_margin);
        }






    }


    //////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////

    public class MultiChartArray_TraceItem
    {
        public string name;
        public int traceNum;
        public int id;

        public MultiChartArray_TraceItem(string Name, int TraceNum, int ID)
        {
            name = Name;
            traceNum = TraceNum;
            id = ID;            
        }
    }


    public class MultiChartArray_Message_EventArgs : EventArgs
    {
        private string message;    

        public MultiChartArray_Message_EventArgs(string _message)
        {
            message = _message;
        }

        public string Message
        {
            get { return message; }
            set { message = value; }
        }
    }


    public class MultiChartArray_RangeChanged_EventArgs : EventArgs
    {
        private int xStart;
        private int xEnd;
        private int yStart;
        private int yEnd;

        public MultiChartArray_RangeChanged_EventArgs(int x1, int y1, int x2, int y2)
        {
            xStart = x1;
            xEnd = x2;
            yStart = y1;
            yEnd = y2;
        }

        public int XStart
        {
            get { return xStart; }
            set { xStart = value; }
        }

        public int XEnd
        {
            get { return xEnd; }
            set { xEnd = value; }
        }

        public int YStart
        {
            get { return yStart; }
            set { yStart = value; }
        }

        public int YEnd
        {
            get { return yEnd; }
            set { yEnd = value; }
        }
    }

}
