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

namespace WPFTools
{
    /// <summary>
    /// Interaction logic for RunningIndicatorControl.xaml
    /// </summary>
    public partial class RunningIndicatorControl : UserControl
    {
        RunningIndicatorControl_ViewModel m_vm;

        public RunningIndicatorControl()
        {
            InitializeComponent();
            m_vm = new RunningIndicatorControl_ViewModel();
            DataContext = m_vm;
        }

        public void IsRunning(bool val)
        {
            m_vm.active = val;
        }
    }


    public class RunningIndicatorControl_ViewModel : INotifyPropertyChanged
    {
        private bool _active;
        public bool active
        {
            get { return _active; }
            set { _active = value; OnPropertyChanged(new PropertyChangedEventArgs("active")); }
        }

        public RunningIndicatorControl_ViewModel()
        {
            active = false;
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

