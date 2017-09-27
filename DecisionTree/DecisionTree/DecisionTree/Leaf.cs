using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree
{
    public class Leaf
    {
        private Leaf output1 = null;
        private Leaf output2 = null;
        private double split;
        private int variable;
        private double nBackground = 0;
        private double nSignal = 0;

        public Leaf() :
            this(0, 0)
        { }

        public Leaf(int variable, double split)
        {
            this.variable = variable;
            this.split = split;
        }

        public void WriteToFile(string filename)
        {
            var bw = new BinaryWriter(File.Create(filename));
            write(bw);
            bw.Close();
        }

        private void write(BinaryWriter bw)
        {
            bw.Write(variable);
            bw.Write(split);

            bool isfinal = IsFinal();
            bw.Write(isfinal);
            if (!isfinal)
            {
                output1.write(bw);
                output2.write(bw);
            }
        }

        public Leaf(string filename)
        {
            var br = new BinaryReader(File.OpenRead(filename));
            read(br);
        }

        private Leaf(BinaryReader sr)
        {
            read(sr);
        }

        private void read(BinaryReader br)
        {
            variable = br.ReadInt32();
            split = br.ReadDouble();
            bool fin = br.ReadBoolean();
            if (!fin)
            {
                output1 = new Leaf(br);
                output2 = new Leaf(br);
            }
        }

        public bool IsFinal()
        {
            return output1 == null || output2 == null;
        }

        public double GetPurity()
        {
            return 1.0*nSignal/ (1.0 * nBackground + nSignal);
        }

        public double RunDataPoint(DataPoint dataPoint)
        {
            if (IsFinal())
            {
                return GetPurity();
            }

            if (doSplit(dataPoint))
            {
                return output1.RunDataPoint(dataPoint);
            }
            else
            {
                return output2.RunDataPoint(dataPoint);
            }
        }

        private bool doSplit(DataPoint dataPoint)
        {
            return dataPoint.Variables[variable] <= split;
        }

        public void Train(DataSet signal, DataSet background)
        {
            foreach (var p in  signal.Points) {
                nSignal = nSignal + p.weight;
            }
            foreach (var p in background.Points)
            {
                nBackground = nBackground + p.weight;
            }
            bool branch = chooseVariable(signal, background);

            if (branch)
            {
                output1 = new Leaf();
                output2 = new Leaf();

                DataSet signalLeft = new DataSet();
                DataSet signalRight = new DataSet();
                DataSet backgroundLeft = new DataSet();
                DataSet backgroundRight = new DataSet();

                foreach (var dataPoint in signal.Points)
                {
                    if (doSplit(dataPoint))
                    {
                        signalLeft.AddDataPoint(dataPoint);
                    }
                    else
                    {
                        signalRight.AddDataPoint(dataPoint);
                    }
                }

                foreach (var dataPoint in background.Points)
                {
                    if (doSplit(dataPoint))
                    {
                        backgroundLeft.AddDataPoint(dataPoint);
                    }
                    else
                    {
                        backgroundRight.AddDataPoint(dataPoint);
                    }
                }

                output1.Train(signalLeft, backgroundLeft);
                output2.Train(signalRight, backgroundRight);
            }
        }

        private bool chooseVariable(DataSet signal, DataSet background)
        {

            int bestVar = -1;
            double bestSplit = -1;
            double bestPurityDifference = -1;
            int nVars = signal.Points[0].Variables.Length;
            for (int i = 0; i < nVars; i++) {
                double maxSignal = Double.MinValue;
                double minSignal = Double.MaxValue;
                double maxBackground = Double.MinValue;
                double minBackground = Double.MaxValue;
                foreach (var p in signal.Points) {

                    if (p.Variables[i] > maxSignal) {
                        maxSignal = p.Variables[i];
                    }
                    if (p.Variables[i] < minSignal)
                    {
                        minSignal = p.Variables[i];
                    }
                }
                foreach (var p in background.Points)
                {
                    if (p.Variables[i] > maxBackground)
                    {
                        maxBackground = p.Variables[i];
                    }
                    if (p.Variables[i] < minBackground)
                    {
                        minBackground = p.Variables[i];
                    }
                }
                double min = Math.Min(minBackground, minSignal);
                double max = Math.Max(maxBackground, maxSignal);

                double increment = Math.Abs((max - min) / 1000);



                for (double j = min; j < max; j = j + increment) {
                    double signalAbove = 0;
                    double signalBelow = 0;
                    double backgroundAbove = 0;
                    double backgroundBelow = 0;
                    foreach (var p in signal.Points) {
                        if (p.Variables[i] > j) {
                            signalAbove = signalAbove + p.weight;
                        }
                        if(p.Variables[i] < j) {
                            signalBelow = signalBelow + p.weight;
                        }
                    }
                    foreach (var p in background.Points)
                    {
                        if (p.Variables[i] > j)
                        {
                            backgroundAbove = backgroundAbove + p.weight;
                        }
                        if(p.Variables[i] < j) {
                            backgroundBelow = backgroundBelow + p.weight;
                        }
                    }


                    double purityAbove = signalAbove / (1.0*signalAbove + signalBelow);
                    double purityBelow = backgroundAbove / (1.0*backgroundBelow + backgroundAbove);
                    

                    if (Math.Abs(purityAbove - purityBelow) > bestPurityDifference && signalAbove > 50 && signalBelow>50 && backgroundBelow> 50 && backgroundAbove>50)
                    {

                        bestSplit = j;
                        bestVar = i;
                        bestPurityDifference = Math.Abs(purityAbove - purityBelow);

                    }

                }


            }
            if (bestPurityDifference > 0)
            {
                split = bestSplit;
                variable = bestVar;
                return true;
            }
            return false;
        }


    }
}
