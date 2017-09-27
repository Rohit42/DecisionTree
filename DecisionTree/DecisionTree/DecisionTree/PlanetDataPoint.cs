using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree
{
    public class PlanetDataPoint : DataPoint
    {
        private const int eventSize = 8;

        private static string[] names = { "Mass", "Radius", "Carbon percentage", "Liquid percentage", "Iron percentage", "Core density", "Oxygen percentage", "Radioactivity" };

        public PlanetDataPoint() :
            base(eventSize)
        { }

        public override string GetName(int index)
        {
            return names[index];
        }
    }
}
