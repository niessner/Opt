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
using System.Threading;

namespace UIWindow
{



    public partial class MainWindow : Form
    {
        DLLInterface dll;
        public string imageFilename;
        bool textChangedSinceLastCompile = false;
        bool textChangedSinceLastTick = false;
        bool compiling = false;
        List<string> optimizationMethods = new List<string> {
            "gradientDescentCPU",
            "gradientDescentGPU",
            "conjugateGradientCPU",
            "linearizedConjugateGradientCPU",
            "lbfgsCPU",
            "vlbfgsCPU",
            "vlbfgsGPU",
            "gaussNewtonGPU"
            };

        public MainWindow()
        {
            InitializeComponent();
            foreach (string method in optimizationMethods)
                optimizationMethodComboBox.Items.Add(method);
            optimizationMethodComboBox.SelectedItem = "gradientDescentGPU";
        }

        private void MainWindow_Load(object sender, EventArgs e)
        {
            dll = new DLLInterface();
            RunApp();
            loadSourceFile("E:/Projects/DSL/Optimization/API/testMLib/shapeFromShadingAD.t");
        }

        private void loadSourceFile(string filename)
        {
            dll.ProcessCommand("load\t" + filename);
            textBoxEnergy.Text = File.ReadAllText(filename);
            Util.formatSourceLine(textBoxEnergy, 0, textBoxEnergy.Text.Length);
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

        private void runOptimizer()
        {
            var optimizationMethod = (string)textBoxEnergy.Invoke(new Func<string>(() => optimizationMethodComboBox.SelectedItem.ToString()));
            compiling = true;
            uint result = dll.ProcessCommand("run\t" + optimizationMethod);
            if (result == 2)
            {
                compiling = false;
            }
        }

        private async Task Compile() // No async because the method does not need await
        {
            
            await Task.Run(() =>
            {
                saveCode();
                runOptimizer();
            });
        }

        public static Task<bool> StartSTATask(Action func)
        {
            var tcs = new TaskCompletionSource<bool>();
            Thread thread = new Thread(() =>
            {
                try
                {
                    func();
                    tcs.SetResult(true);
                }
                catch (Exception e)
                {
                    tcs.SetException(e);
                }
            });
            thread.SetApartmentState(ApartmentState.STA);
            thread.Start();
            return tcs.Task;
        }
        private Task<bool> RunApp()
        {
            return StartSTATask(() =>
            {
                dll.RunApp();
            });
        }

        /*
        private async Task RunApp() // No async because the method does not need await
        {
            await Task.Run(() =>
            {
                dll.RunApp();
            });
        }*/

        private void compileButton_Click(object sender, EventArgs e)
        {
            if (!compiling)
            {
                Compile();
            }
        }

        private void checkStatus()
        {
            UInt32 status = dll.ProcessCommand("check");
            if (status == 1)
            {
                string errorString = dll.GetString("error");
                textBoxCompiler.Invoke(new Action<string>(s => textBoxCompiler.Text = s), errorString);
                defineTimeBox.Invoke(new Action<string>(s => defineTimeBox.Text = s), dll.GetFloat("defineTime").ToString() + "ms");
                planTimeBox.Invoke(new Action<string>(s => planTimeBox.Text = s), dll.GetFloat("planTime").ToString() + "ms");
                cpuSolveTimeBox.Invoke(new Action<string>(s => cpuSolveTimeBox.Text = s), dll.GetFloat("solveTime").ToString() + "ms");
                gpuSolveTimeBox.Invoke(new Action<string>(s => gpuSolveTimeBox.Text = s), dll.GetFloat("solveTimeGPU").ToString() + "ms");
                compiling = false;
            }
        }

        private void timer_Tick(object sender, EventArgs e)
        {

            checkStatus();
            if (textChangedSinceLastCompile && !textChangedSinceLastTick && !compiling)
            {
                Compile();
                textChangedSinceLastCompile = false;
            }
            textChangedSinceLastTick = false;
        }

        private void textBoxEnergy_TextChanged(object sender, EventArgs e)
        {
            textChangedSinceLastCompile = true;
            textChangedSinceLastTick = true;
        }

        private void fastTimer_Tick(object sender, EventArgs e)
        {
            Point location = pictureBoxMinimum.PointToScreen(Point.Empty);
            dll.MoveWindow(location.X, location.Y, pictureBoxMinimum.Width, pictureBoxMinimum.Height);
        }

    

    }
}
