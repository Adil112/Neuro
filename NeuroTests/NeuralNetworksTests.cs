using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuro.Tests
{
    [TestClass()]
    public class NeuralNetworksTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var dataset = new List<Tuple<double, double[]>>
            {
                /*  2 входных нейрона месяц и день
                 *  1 скрытый слой c 2 нейронами
                 *  1 выходной
                 *  месяц задается как (номер месяца по порядку/12)
                 *  день задается как (номер дня по порядку/29-31) в зависмости от месяца
                 *  знак зодиака соответсвенно (знако зодиака по порядку/12)
                 */
                //                                                M       D
                // водолей
                new Tuple<double, double[]> (1/12, new double[] { 1/12, 20/31}),
                new Tuple<double, double[]> (1/12, new double[] { 2/12, 3/28}),
                new Tuple<double, double[]> (1/12, new double[] { 2/12, 19/28}),
                //рыбы
                new Tuple<double, double[]> (2/12, new double[] { 2/12, 20/28}),
                new Tuple<double, double[]> (2/12, new double[] { 3/12, 5/31}),
                new Tuple<double, double[]> (2/12, new double[] { 3/12, 20/31}),
                //овен
                new Tuple<double, double[]> (3/12, new double[] { 3/12, 21/31}),
                new Tuple<double, double[]> (3/12, new double[] { 4/12, 5/30}),
                new Tuple<double, double[]> (3/12, new double[] { 4/12, 19/30}),
                //телец
                new Tuple<double, double[]> (4/12, new double[] { 4/12, 20/30}),
                new Tuple<double, double[]> (4/12, new double[] { 5/12, 5/31}),
                new Tuple<double, double[]> (4/12, new double[] { 5/12, 20/31}),
                //близнецы
                new Tuple<double, double[]> (5/12, new double[] { 5/12, 21/31}),
                new Tuple<double, double[]> (5/12, new double[] { 6/12, 5/30}),
                new Tuple<double, double[]> (5/12, new double[] { 6/12, 20/30}),
                //рак
                new Tuple<double, double[]> (6/12, new double[] { 6/12, 21/30}),
                new Tuple<double, double[]> (6/12, new double[] { 7/12, 6/31}),
                new Tuple<double, double[]> (6/12, new double[] { 7/12, 22/31}),
                //лев
                new Tuple<double, double[]> (7/12, new double[] { 7/12, 23/31}),
                new Tuple<double, double[]> (7/12, new double[] { 8/12, 7/31}),
                new Tuple<double, double[]> (7/12, new double[] { 8/12, 22/31}),
                //дева
                new Tuple<double, double[]> (8/12, new double[] { 8/12, 23/31}),
                new Tuple<double, double[]> (8/12, new double[] { 9/12, 8/30}),
                new Tuple<double, double[]> (8/12, new double[] { 9/12, 22/30}),
                //весы
                new Tuple<double, double[]> (9/12, new double[] { 9/12, 23/30}),
                new Tuple<double, double[]> (9/12, new double[] { 10/12, 8/31}),
                new Tuple<double, double[]> (9/12, new double[] { 10/12, 23/31}),
                //скорпион  
                 new Tuple<double, double[]> (10/12, new double[] { 10/12, 24/31}),
                new Tuple<double, double[]> (10/12, new double[] { 11/12, 8/30}),
                new Tuple<double, double[]> (10/12, new double[] { 11/12, 22/30}),
                //стрелец
                new Tuple<double, double[]> (11/12, new double[] { 11/12, 23/30}),
                new Tuple<double, double[]> (11/12, new double[] { 12/12, 8/31}),
                new Tuple<double, double[]> (11/12, new double[] { 12/12, 21/31}),
                //козерог
                new Tuple<double, double[]> (12/12, new double[] { 12/12, 21/31}),
                new Tuple<double, double[]> (12/12, new double[] { 1/12, 7/31}),
                new Tuple<double, double[]> (12/12, new double[] { 1/12, 19/31})
            };
            var topology = new Topology(2, 1, 0.1, 2, 2);
            var neuralNetwork = new NeuralNetworks(topology);
            var difference = neuralNetwork.Learn(dataset, 1000); // обучаем

            var results = new List<double>();
            foreach(var data in dataset)
            {
                results.Add(neuralNetwork.FeedForward(data.Item2).Output); // используем
            }

            for(int i =0; i < results.Count; i++)
            {
                var expected = Math.Round(dataset[i].Item1, 4);
                var actual = Math.Round(results[i], 4);
                //Assert.AreEqual(expected, actual);
            }
        }
    }
}