using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.IO;
using System.Net;
using System.Linq;
using System.Diagnostics;
using System.Windows.Input;

namespace InteractiveApp
{
    [StructLayout(LayoutKind.Sequential), Serializable]
    public struct IVBitmapInfo
    {
        [MarshalAs(UnmanagedType.U4)]
        public int width;
        [MarshalAs(UnmanagedType.U4)]
        public int height;
        [MarshalAs(UnmanagedType.SysInt)]
        public IntPtr colorData;
    }

    public class DLLInterface
    {
        const string InteractiveDLL = "Interactive.dll";
        [DllImport(InteractiveDLL, CharSet = CharSet.Unicode)]
        public static extern IntPtr IVInit();
        /*[DllImport(InteractiveDLL)]
        private static extern UInt32 IVProcessCommand(IntPtr context, [In, MarshalAs(UnmanagedType.LPStr)] String command);
        [DllImport(InteractiveDLL)]
        public static extern IntPtr IVGetBitmapByName(IntPtr context, [In, MarshalAs(UnmanagedType.LPStr)] String bitmapName);
        [DllImport(InteractiveDLL)]
        public static extern IntPtr IVGetStringByName(IntPtr context, [In, MarshalAs(UnmanagedType.LPStr)] String stringName);
        [DllImport(InteractiveDLL)]
        public static extern Int32 IVGetIntegerByName(IntPtr context, [In, MarshalAs(UnmanagedType.LPStr)] String integerName);*/
        
        public IntPtr interactiveDLLContext = (IntPtr)0;

        public DLLInterface()
        {
            if (interactiveDLLContext == (IntPtr)0)
            {
                interactiveDLLContext = IVInit();
            }
        }

        /*public UInt32 ProcessCommand(String command)
        {
            return IVProcessCommand(interactiveDLLContext, command);
        }

        public Bitmap GetBitmap(String bitmapName)
        {
            IntPtr bitmapInfoUnmanaged = IVGetBitmapByName(interactiveDLLContext, bitmapName);
            if (bitmapInfoUnmanaged == (IntPtr)0) return null;

            IVBitmapInfo bitmapInfo = (IVBitmapInfo)Marshal.PtrToStructure(bitmapInfoUnmanaged, typeof(IVBitmapInfo));
            if (bitmapInfo.width == 0 || bitmapInfo.height == 0 || bitmapInfo.colorData == null) return null;

            return new Bitmap(bitmapInfo.width, bitmapInfo.height, bitmapInfo.width * 4, System.Drawing.Imaging.PixelFormat.Format32bppRgb, bitmapInfo.colorData);
        }

        public String GetString(String stringName)
        {
            IntPtr stringPtr = IVGetStringByName(interactiveDLLContext, stringName);
            if (stringPtr == (IntPtr)0)
            {
                return null;
            }
            else
            {
                return Marshal.PtrToStringAnsi(stringPtr);
            }
        }*/
    }
}
