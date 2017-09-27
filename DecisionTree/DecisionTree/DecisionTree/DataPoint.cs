using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree
{
    abstract public class DataPoint
    {
        public double[] Variables { get; set; }
        public double weight { get; set; }
        public double averageSum { get; set; }
        
        public DataPoint(int nvar)
        {
            Variables = new double[nvar];
        }

        abstract public string GetName(int index);

        public void WriteToFile(BinaryWriter file)
        {
            string type = GetType().ToString();
            file.Write(type);

            foreach (var variable in Variables)
            {
                file.Write(variable);
            }
        }

        static public DataPoint ReadFromFile(BinaryReader file) 
        {
            
            string name = file.ReadString();
            Assembly assembly = typeof(DataPoint).Assembly;
            Type type = assembly.GetType(name);
            DataPoint answer = Activator.CreateInstance(type) as DataPoint;

            for (int i = 0; i < answer.Variables.Length; ++i)
            {
                answer.Variables[i] = file.ReadDouble();
            }

            return answer;
        }
    }
}
