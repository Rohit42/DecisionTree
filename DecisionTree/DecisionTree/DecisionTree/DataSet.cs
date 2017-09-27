using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree
{
    public class DataSet
    {
        public IList<DataPoint> Points { get; set; } = new List<DataPoint>();

        public DataSet()
        { }

        public void AddDataPoint(DataPoint point)
        {
            Points.Add(point);
        }

        public void WriteToFile(string filename)
        {
            var file = new BinaryWriter(File.Create(filename));
            file.Write(Points.Count);
            foreach(var point in Points)
            {
                point.WriteToFile(file);
            }
            file.Close();
        }

        public DataSet(string filename)
        {
            var file = new BinaryReader(File.OpenRead(filename));
            int size = file.ReadInt32();

            for (int i = 0; i < size; ++i)
            {
                DataPoint dp = DataPoint.ReadFromFile(file);
                AddDataPoint(dp);
            }

            file.Close();
        }
    }
}
