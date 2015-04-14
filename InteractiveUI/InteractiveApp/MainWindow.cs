using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using InteractiveApp;

namespace UIWindow
{
    public partial class MainWindow : Form
    {
        DLLInterface dll;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void MainWindow_Load(object sender, EventArgs e)
        {
            dll = new DLLInterface();
        }

    }
}
