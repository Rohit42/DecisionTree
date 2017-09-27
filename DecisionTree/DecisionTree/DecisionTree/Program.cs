using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree
{
    class Program
    {
        static void Main(string[] args)
        {
            DataSet ds = new DataSet(@"C:\Users\Rohit\Documents\Visual Studio 2017\Projects\DecisionTree\DecisionTree\DecisionTree\signal.dat");
            DataSet db = new DataSet(@"C:\Users\Rohit\Documents\Visual Studio 2017\Projects\DecisionTree\DecisionTree\DecisionTree\background.dat");
            DataSet data = new DataSet(@"C:\Users\Rohit\Documents\Visual Studio 2017\Projects\DecisionTree\DecisionTree\DecisionTree\decisionTreeData.dat");
            int treeCount = 11;
            double[] weights = new double[treeCount+1];
            weights[0] = 1;
            Leaf[] forest = new Leaf[treeCount];
            for (int i = 0; i < ds.Points.Count; i++)
            {
                ds.Points[i].weight = 1 ;

            }
            for (int i = 0; i < db.Points.Count; i++)
            {
                db.Points[i].weight = 1;
            }
            for (int h = 0; h < treeCount; h++)
            {
                Console.WriteLine(h);
                forest[h] = new Leaf();
                forest[h].Train(ds, db);
                Console.WriteLine("done training");
                double counter = 0;
                for (int i = 0; i < ds.Points.Count; i++)
                {
                    double r = forest[h].RunDataPoint(ds.Points[i]);
                    
                    if (r < 0.5)
                    {
                        counter++;

                    }
                }
                for (int i = 0; i < db.Points.Count; i++)
                {
                    double r = forest[h].RunDataPoint(db.Points[i]);
                    if (r > 0.5)
                    {
                        counter++;
                    }
                }
                counter = counter / (ds.Points.Count + db.Points.Count);
                weights[h + 1] = (1-counter) / counter;
                for (int i = 0; i < ds.Points.Count; i++)
                {
                    double r = forest[h].RunDataPoint(ds.Points[i]);
                    if (r < 0.5)
                    {
                        double weight = (1 - counter) / counter;

                        ds.Points[i].weight = weight;
                    }
                    else {
                        ds.Points[i].weight = 1;
                    }
                }
          
                for (int i = 0; i < db.Points.Count; i++)
                {
                    double r = forest[h].RunDataPoint(db.Points[i]);
                    if (r > 0.5)
                    {
                        double weight = (1 - counter) / counter;

                        db.Points[i].weight = weight;
                    }
                    else
                    {
                        db.Points[i].weight = 1;
                    }
                }
            }
            foreach (var d in data.Points) {
                for (int i = 0; i < treeCount; i++) {
                    var tree = forest[i];
                    double r = tree.RunDataPoint(d);
                    Console.WriteLine(d.averageSum);

                    d.averageSum = d.averageSum + Math.Log(weights[i])*r;
                }
            }
            using (System.IO.StreamWriter file = new System.IO.StreamWriter(@"C:\Users\Rohit\Documents\Visual Studio 2017\Projects\DecisionTree\DecisionTree\DecisionTree\finaldata.csv"))
                for (int i = 0; i < data.Points.Count; i++) {
                file.WriteLine(i + "," + data.Points[i].averageSum);
                    
            }
            Console.ReadKey();

        }
    }
}
