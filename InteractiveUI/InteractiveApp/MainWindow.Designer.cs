namespace UIWindow
{
    partial class MainWindow
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.textBoxEnergy = new System.Windows.Forms.RichTextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.textBoxCompiler = new System.Windows.Forms.RichTextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.pictureBoxMinimum = new System.Windows.Forms.PictureBox();
            this.buttonLoad = new System.Windows.Forms.Button();
            this.Save = new System.Windows.Forms.Button();
            this.LoadImage = new System.Windows.Forms.Button();
            this.compileButton = new System.Windows.Forms.Button();
            this.resultImageBox = new System.Windows.Forms.PictureBox();
            this.timer = new System.Windows.Forms.Timer(this.components);
            this.optimizationMethodComboBox = new System.Windows.Forms.ComboBox();
            this.label4 = new System.Windows.Forms.Label();
            this.defineTimeBox = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.planTimeBox = new System.Windows.Forms.TextBox();
            this.cpuSolveTimeBox = new System.Windows.Forms.TextBox();
            this.gpuSolveTimeBox = new System.Windows.Forms.TextBox();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxMinimum)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.resultImageBox)).BeginInit();
            this.SuspendLayout();
            // 
            // textBoxEnergy
            // 
            this.textBoxEnergy.Font = new System.Drawing.Font("Consolas", 12F);
            this.textBoxEnergy.Location = new System.Drawing.Point(13, 39);
            this.textBoxEnergy.Margin = new System.Windows.Forms.Padding(4);
            this.textBoxEnergy.Name = "textBoxEnergy";
            this.textBoxEnergy.Size = new System.Drawing.Size(680, 536);
            this.textBoxEnergy.TabIndex = 0;
            this.textBoxEnergy.Text = "";
            this.textBoxEnergy.TextChanged += new System.EventHandler(this.textBoxEnergy_TextChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(13, 9);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(121, 19);
            this.label1.TabIndex = 1;
            this.label1.Text = "Energy function:";
            // 
            // textBoxCompiler
            // 
            this.textBoxCompiler.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.textBoxCompiler.Location = new System.Drawing.Point(13, 639);
            this.textBoxCompiler.Margin = new System.Windows.Forms.Padding(4);
            this.textBoxCompiler.Name = "textBoxCompiler";
            this.textBoxCompiler.ReadOnly = true;
            this.textBoxCompiler.Size = new System.Drawing.Size(518, 148);
            this.textBoxCompiler.TabIndex = 0;
            this.textBoxCompiler.Text = "";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(9, 616);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(126, 19);
            this.label2.TabIndex = 1;
            this.label2.Text = "Compiler output:";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(637, 9);
            this.label3.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(175, 19);
            this.label3.TabIndex = 1;
            this.label3.Text = "Minimum energy image:";
            // 
            // pictureBoxMinimum
            // 
            this.pictureBoxMinimum.Location = new System.Drawing.Point(700, 39);
            this.pictureBoxMinimum.Name = "pictureBoxMinimum";
            this.pictureBoxMinimum.Size = new System.Drawing.Size(628, 382);
            this.pictureBoxMinimum.TabIndex = 2;
            this.pictureBoxMinimum.TabStop = false;
            // 
            // buttonLoad
            // 
            this.buttonLoad.Location = new System.Drawing.Point(153, 4);
            this.buttonLoad.Name = "buttonLoad";
            this.buttonLoad.Size = new System.Drawing.Size(144, 28);
            this.buttonLoad.TabIndex = 3;
            this.buttonLoad.Text = "Load Source";
            this.buttonLoad.UseVisualStyleBackColor = true;
            this.buttonLoad.Click += new System.EventHandler(this.buttonLoad_Click);
            // 
            // Save
            // 
            this.Save.Location = new System.Drawing.Point(303, 4);
            this.Save.Name = "Save";
            this.Save.Size = new System.Drawing.Size(150, 28);
            this.Save.TabIndex = 5;
            this.Save.Text = "Save Source";
            this.Save.UseVisualStyleBackColor = true;
            this.Save.Click += new System.EventHandler(this.buttonSave_Click);
            // 
            // LoadImage
            // 
            this.LoadImage.Location = new System.Drawing.Point(819, 5);
            this.LoadImage.Name = "LoadImage";
            this.LoadImage.Size = new System.Drawing.Size(150, 28);
            this.LoadImage.TabIndex = 6;
            this.LoadImage.Text = "Load Image";
            this.LoadImage.UseVisualStyleBackColor = true;
            this.LoadImage.Click += new System.EventHandler(this.buttonLoadImage_Click);
            // 
            // compileButton
            // 
            this.compileButton.Location = new System.Drawing.Point(12, 582);
            this.compileButton.Name = "compileButton";
            this.compileButton.Size = new System.Drawing.Size(149, 26);
            this.compileButton.TabIndex = 7;
            this.compileButton.Text = "Compile and Run";
            this.compileButton.UseVisualStyleBackColor = true;
            this.compileButton.Click += new System.EventHandler(this.compileButton_Click);
            // 
            // resultImageBox
            // 
            this.resultImageBox.Location = new System.Drawing.Point(700, 427);
            this.resultImageBox.Name = "resultImageBox";
            this.resultImageBox.Size = new System.Drawing.Size(628, 360);
            this.resultImageBox.TabIndex = 8;
            this.resultImageBox.TabStop = false;
            // 
            // timer
            // 
            this.timer.Enabled = true;
            this.timer.Interval = 200;
            this.timer.Tick += new System.EventHandler(this.timer_Tick);
            // 
            // optimizationMethodComboBox
            // 
            this.optimizationMethodComboBox.FormattingEnabled = true;
            this.optimizationMethodComboBox.Location = new System.Drawing.Point(263, 581);
            this.optimizationMethodComboBox.Name = "optimizationMethodComboBox";
            this.optimizationMethodComboBox.Size = new System.Drawing.Size(369, 27);
            this.optimizationMethodComboBox.TabIndex = 9;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(538, 639);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(57, 19);
            this.label4.TabIndex = 10;
            this.label4.Text = "Define:";
            // 
            // defineTimeBox
            // 
            this.defineTimeBox.Location = new System.Drawing.Point(602, 631);
            this.defineTimeBox.Name = "defineTimeBox";
            this.defineTimeBox.Size = new System.Drawing.Size(92, 27);
            this.defineTimeBox.TabIndex = 11;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(538, 671);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(43, 19);
            this.label5.TabIndex = 12;
            this.label5.Text = "Plan:";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(538, 701);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(46, 19);
            this.label6.TabIndex = 13;
            this.label6.Text = "Solve";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(547, 720);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(40, 19);
            this.label7.TabIndex = 14;
            this.label7.Text = "CPU:";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(547, 755);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(42, 19);
            this.label8.TabIndex = 15;
            this.label8.Text = "GPU:";
            // 
            // planTimeBox
            // 
            this.planTimeBox.Location = new System.Drawing.Point(602, 665);
            this.planTimeBox.Name = "planTimeBox";
            this.planTimeBox.Size = new System.Drawing.Size(92, 27);
            this.planTimeBox.TabIndex = 16;
            // 
            // cpuSolveTimeBox
            // 
            this.cpuSolveTimeBox.Location = new System.Drawing.Point(602, 718);
            this.cpuSolveTimeBox.Name = "cpuSolveTimeBox";
            this.cpuSolveTimeBox.Size = new System.Drawing.Size(92, 27);
            this.cpuSolveTimeBox.TabIndex = 17;
            // 
            // gpuSolveTimeBox
            // 
            this.gpuSolveTimeBox.Location = new System.Drawing.Point(602, 752);
            this.gpuSolveTimeBox.Name = "gpuSolveTimeBox";
            this.gpuSolveTimeBox.Size = new System.Drawing.Size(92, 27);
            this.gpuSolveTimeBox.TabIndex = 18;
            // 
            // MainWindow
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 19F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1340, 795);
            this.Controls.Add(this.gpuSolveTimeBox);
            this.Controls.Add(this.cpuSolveTimeBox);
            this.Controls.Add(this.planTimeBox);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.defineTimeBox);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.optimizationMethodComboBox);
            this.Controls.Add(this.resultImageBox);
            this.Controls.Add(this.compileButton);
            this.Controls.Add(this.LoadImage);
            this.Controls.Add(this.Save);
            this.Controls.Add(this.buttonLoad);
            this.Controls.Add(this.pictureBoxMinimum);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.textBoxCompiler);
            this.Controls.Add(this.textBoxEnergy);
            this.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Margin = new System.Windows.Forms.Padding(4);
            this.Name = "MainWindow";
            this.Text = "Interactive Optimizer";
            this.Load += new System.EventHandler(this.MainWindow_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxMinimum)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.resultImageBox)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.RichTextBox textBoxEnergy;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.RichTextBox textBoxCompiler;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.PictureBox pictureBoxMinimum;
        private System.Windows.Forms.Button buttonLoad;
        private System.Windows.Forms.Button Save;
        private System.Windows.Forms.Button LoadImage;
        private System.Windows.Forms.Button compileButton;
        private System.Windows.Forms.PictureBox resultImageBox;
        private System.Windows.Forms.Timer timer;
        private System.Windows.Forms.ComboBox optimizationMethodComboBox;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox defineTimeBox;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox planTimeBox;
        private System.Windows.Forms.TextBox cpuSolveTimeBox;
        private System.Windows.Forms.TextBox gpuSolveTimeBox;

    }
}

