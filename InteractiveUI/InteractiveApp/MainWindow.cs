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
        public string imageFilename;
        bool textChangedSinceLastCompile = false;
        bool compiling = false;
        List<string> optimizationMethods = new List<string> {
            "gradientDescentCPU",
            "gradientDescentGPU",
            "conjugateGradientCPU",
            "linearizedConjugateGradientCPU",
            "linearizedConjugateGradientGPU", 
            "linearizedPreconditionedConjugateGradientCPU",
            "lbfgsCPU"};

        public MainWindow()
        {
            InitializeComponent();
            foreach (string method in optimizationMethods)
                optimizationMethodComboBox.Items.Add(method);
            optimizationMethodComboBox.SelectedItem = "linearizedPreconditionedConjugateGradientCPU";
        }

        private void MainWindow_Load(object sender, EventArgs e)
        {
            dll = new DLLInterface();
        }

        private void loadSourceFile(string filename)
        {
            dll.ProcessCommand("load\t" + filename);
            textBoxEnergy.Text = File.ReadAllText(filename);
            Util.formatSourceLine(textBoxEnergy, 0, textBoxEnergy.Text.Length);
        }

        private void loadImageFile(string filename)
        {
            imageFilename = filename;
            // TODO: Extend to handle multiple images
            dll.ProcessCommand("loadImage\t" + imageFilename);
            pictureBoxMinimum.Image = dll.GetBitmap("test");
            resultImageBox.Image = dll.GetBitmap("result");
        }

        private void buttonLoad_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();

            dialog.Filter = "Terra Files (.t)|*.t|All Files (*.*)|*.*";
            dialog.FilterIndex = 1;

            var relativePath =  @"\..\..\API\testMlib";
            dialog.InitialDirectory = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory() + relativePath));

            dialog.Multiselect = false;

            DialogResult result = dialog.ShowDialog();

            if (result == DialogResult.OK)
            {
                loadSourceFile(dialog.FileName);
            }
        }

        private void saveCode()
        {
            var text = (string)textBoxEnergy.Invoke(new Func<string>(() => textBoxEnergy.Text));
            File.WriteAllText(dll.GetString("terraFilename"), text);
        }

        private void buttonSave_Click(object sender, EventArgs e)
        {
            saveCode();
        }

        private void buttonLoadImage_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();

            dialog.Filter = "image Files (.png)|*.png|All Files (*.*)|*.*";
            dialog.FilterIndex = 1;

            var relativePath = @"\..\..\API\testMlib";
            dialog.InitialDirectory = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory() + relativePath));

            dialog.Multiselect = false;

            DialogResult result = dialog.ShowDialog();

            if (result == DialogResult.OK)
            {
                loadImageFile(dialog.FileName);
            }
        }

        private void runOptimizer()
        {
            var optimizationMethod = (string)textBoxEnergy.Invoke(new Func<string>(() => optimizationMethodComboBox.SelectedItem.ToString()));
            
            uint result = dll.ProcessCommand("run\t" + optimizationMethod);
            if (result != 0)
            {
                string errorString = dll.GetString("error");
                textBoxCompiler.Invoke(new Action<string>(s => textBoxCompiler.Text = s), errorString);
            }
            else
            {
                textBoxCompiler.Invoke(new Action<string>(s => textBoxCompiler.Text = s), "Success!");
                textBoxCompiler.Invoke(new Action<Bitmap>(b => resultImageBox.Image = b), dll.GetBitmap("result"));
            }
        }

        private async Task Compile() // No async because the method does not need await
        {
            compiling = true;
            await Task.Run(() =>
            {
                saveCode();
                runOptimizer();
            });
            compiling = false;
        }

        private void compileButton_Click(object sender, EventArgs e)
        {
            if (!compiling)
            {
                Compile();
            }
        }

        private void timer_Tick(object sender, EventArgs e)
        {
            if (textChangedSinceLastCompile && !compiling)
            {
                Compile();
                textChangedSinceLastCompile = false;
            }
        }

        private void textBoxEnergy_TextChanged(object sender, EventArgs e)
        {
            textChangedSinceLastCompile = true;
        }

    

    }
}
