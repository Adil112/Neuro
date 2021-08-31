using System;
using System.Collections.Generic;


namespace Neuro
{


    class Program
    {
        static void Main(string[] args)
        {
            


            var dataset = new List<Tuple<double, double[]>>
            {
                new Tuple<double, double[]> (0.01, new double[] { 0.1, 0.1, 0.1}),
                new Tuple<double, double[]> (0.006, new double[] { 0.1, 0.2, 0.3}),
                new Tuple<double, double[]> (0.175, new double[] { 0.5, 0.7, 0.5}),
                new Tuple<double, double[]> (0.012, new double[] { 0.6, 0.1, 0.2}),
                new Tuple<double, double[]> (0.009, new double[] { 0.1, 0.3, 0.3}),
                new Tuple<double, double[]> (0.004, new double[] { 0.1, 0.4, 0.1}),
                new Tuple<double, double[]> (0.0255, new double[] { 0.51, 0.5, 0.1}),
                new Tuple<double, double[]> (0.0246, new double[] { 0.41, 0.6, 0.1}),
                new Tuple<double, double[]> (0.0518, new double[] { 0.1, 0.7, 0.74}),
                new Tuple<double, double[]> (0.3172, new double[] { 0.61, 0.8, 0.65}),
                new Tuple<double, double[]> (0.0639, new double[] { 0.71, 0.9, 0.1}),
                new Tuple<double, double[]> (0.0405, new double[] { 0.1, 0.9, 0.45}),
                new Tuple<double, double[]> (0.0054, new double[] { 0.1, 0.1, 0.54}),
                new Tuple<double, double[]> (0.02904, new double[] { 0.8, 0.11, 0.33}),
                new Tuple<double, double[]> (0.0108, new double[] { 0.1, 0.12, 0.9})
            };
            

            var topology = new Topology(3, 1, 0.3, 2, 2);
            var neuralNetwork = new NeuralNetworks(topology);
            var difference = neuralNetwork.Learn(dataset, 500); // обучаем

            var results = new List<double>();
            foreach (var data in dataset)
            {
                results.Add(neuralNetwork.FeedForward(data.Item2).Output); // используем
            }
            double res = 0;
            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(dataset[i].Item1, 4);
                var actual = Math.Round(results[i], 4);
                double promres = (Math.Pow(expected - actual, 2) / expected) * 100;
                res += promres;
            }
            Console.WriteLine($"Общая погрешность: {res / results.Count}");
            Console.ReadKey();
        }
    }
}
