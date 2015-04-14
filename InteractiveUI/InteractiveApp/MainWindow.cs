using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;

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

        private void loadSourceFile(string filename)
        {
            textBoxEnergy.Text = File.ReadAllText(filename);
            Util.formatSourceLine(textBoxEnergy, 0, textBoxEnergy.Text.Length);
        }

        private void buttonLoad_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();

            dialog.Filter = "Terra Files (.t)|*.t|All Files (*.*)|*.*";
            dialog.FilterIndex = 1;

            dialog.InitialDirectory = @"C:\Code\Opt\Optimization\API\testMLib\";

            dialog.Multiselect = false;

            DialogResult result = dialog.ShowDialog();

            if (result == DialogResult.OK)
            {
                loadSourceFile(dialog.FileName);
            }
        }

    }
}
