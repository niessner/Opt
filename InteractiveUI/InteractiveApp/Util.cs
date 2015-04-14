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
using System.Text.RegularExpressions;

namespace UIWindow
{
    class Util
    {
        static string[] terraKeywords = {"and", "break", "do", "else", "elseif","end", "false", "for", "function", "goto", "if","in",
                                         "local", "nil", "not", "or", "repeat","return", "then", "true", "until", "while", "terra",
                                         "var","struct","union","quote","import","defer","escape"};
        static Regex keywordRegex;

        public static void makeRegexps()
        {
            if(keywordRegex == null)
            {
                string keywordList = "(";
                foreach(string keyword in terraKeywords)
                {
                    keywordList = keywordList + keyword;
                    if (keyword != terraKeywords.Last())
                        keywordList = keywordList + "|";
                }
                keywordList = keywordList + ")";

                keywordRegex = new Regex("(^|\\s)" + keywordList + "(\\s|^)");
            }
        }

        public static void formatSourceLine(RichTextBox rtb, int start, int length)
        {
            makeRegexps();

            string s = rtb.Text.Substring(start, length);

            //
            // keywords
            //
            foreach (Match match in keywordRegex.Matches(rtb.Text))
            {
                rtb.Select(match.Index, match.Length);
                rtb.SelectionColor = Color.Blue;
            }
        }
    }
}
