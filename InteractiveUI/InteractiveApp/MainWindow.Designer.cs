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
            this.textBoxEnergy = new System.Windows.Forms.RichTextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.textBoxCompiler = new System.Windows.Forms.RichTextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.pictureBoxMinimum = new System.Windows.Forms.PictureBox();
            this.buttonLoad = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxMinimum)).BeginInit();
            this.SuspendLayout();
            // 
            // textBoxEnergy
            // 
            this.textBoxEnergy.Font = new System.Drawing.Font("Consolas", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxEnergy.Location = new System.Drawing.Point(13, 39);
            this.textBoxEnergy.Margin = new System.Windows.Forms.Padding(4);
            this.textBoxEnergy.Name = "textBoxEnergy";
            this.textBoxEnergy.Size = new System.Drawing.Size(619, 536);
            this.textBoxEnergy.TabIndex = 0;
            this.textBoxEnergy.Text = "";
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
            this.textBoxCompiler.Location = new System.Drawing.Point(13, 608);
            this.textBoxCompiler.Margin = new System.Windows.Forms.Padding(4);
            this.textBoxCompiler.Name = "textBoxCompiler";
            this.textBoxCompiler.ReadOnly = true;
            this.textBoxCompiler.Size = new System.Drawing.Size(619, 179);
            this.textBoxCompiler.TabIndex = 0;
            this.textBoxCompiler.Text = "";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(9, 585);
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
            this.pictureBoxMinimum.Location = new System.Drawing.Point(641, 39);
            this.pictureBoxMinimum.Name = "pictureBoxMinimum";
            this.pictureBoxMinimum.Size = new System.Drawing.Size(687, 536);
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
            // MainWindow
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 19F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1340, 795);
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

    }
}

