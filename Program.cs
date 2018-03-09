/*
 * Created by SharpDevelop.
 * User: Benny
 * Date: 2017-01-08
 * Time: 6:36 PM
 * 
 * To change this template use Tools | Options | Coding | Edit Standard Headers.
 */
using System;

using System.Collections.Generic;
	
namespace ann
{
	class Program
	{		
		public static void Main(string[] args)
		{
			Neural_Network ann = new Neural_Network(2,2,1);
			double[][] tdinput = new double[][]
			{
				new double[] {0,0},
				new double[] {0,1},
				new double[] {1,0},
				new double[] {1,1}
			};
			double[][] tdoutput = new double[][]
			{
				new double[] {0},
				new double[] {1},
				new double[] {1},
				new double[] {0}
			};
			
			ann.train(tdinput, tdoutput, 50000);
			
			Console.WriteLine();
			ann.compute(new double[] {1,1});
			
			Console.Write("Press any key to continue . . . ");
			Console.ReadKey(true);
		}
	}
	
	class Neural_Network 
	{
		public List<Layer> layer = new List<Layer>();
		
		private double learning_rate = 0.03;
		
		public Random rnd = new Random();
		
		public Neural_Network(params int[] layout)
		{
			Array.Resize(ref layout, layout.Length + 1);
			for (int i = 0; i < layout.Length - 1; i ++)
			{
				layer.Add(new Layer(layout[i], layout[i + 1]));
			}
		}
		
		private struct Activation_Function
		{
			public double hyperbolic_tangent(double input)
			{
				return 2 / (1 + Math.Exp(-2 * input)) - 1;
			}
			public double hyperbolic_tangent_derivative(double input)
			{
				return 1 - Math.Pow(hyperbolic_tangent(input), 2);
			}
		}
		
		private void Feed_Forward(double[] input)
		{
			Activation_Function af = new Activation_Function();
			
			for (int i = 0; i < layer.Count; i ++)
			{
				if (i == 0) { for (int j = 0; j < layer[i].neuron.Count; j ++) { layer[i].neuron[j].output = input[j]; }}
				else
				{
					for (int j = 0; j < layer[i].neuron.Count; j ++) 
					{
						double weighted_sum = 0;
						for (int k = 0; k < layer[i - 1].neuron.Count; k ++) 
						{
							weighted_sum += layer[i - 1].neuron[k].output * layer[i - 1].neuron[k].weight[j];
						}
						layer[i].neuron[j].output = af.hyperbolic_tangent(weighted_sum + layer[i - 1].bias.output);
					}
				}
			}
		}
		
		private void Back_Propagate(double[] input)
		{
			Activation_Function af = new Activation_Function();
			
			for (int i = layer.Count; i > 0; i --)
			{
				if (i == layer.Count) { for (int j = 0; j < layer[i - 1].neuron.Count; j ++) 
					{ layer[i - 1].neuron[j].error = (input[j] - layer[i - 1].neuron[j].output) *  af.hyperbolic_tangent_derivative(layer[i - 1].neuron[j].output); }}
				else 
				{
					for (int j = 0; j < layer[i - 1].neuron.Count; j ++)
					{
						double error_weight = 0;
						for (int k = 0; k < layer[i].neuron.Count; k ++)
						{
							error_weight += layer[i - 1].neuron[j].weight[k] * layer[i].neuron[k].error;
						}
						layer[i - 1].neuron[j].error = error_weight * af.hyperbolic_tangent_derivative(layer[i - 1].neuron[j].output);
					}
				}
				
			}
		}
		
		private void Update_weights()
		{
			for (int i = 0; i < layer.Count - 1; i ++)
			{
				for (int j = 0; j < layer[i].neuron.Count; j ++)
				{
					for (int k = 0; k < layer[i + 1].neuron.Count; k ++)
					{
						layer[i].neuron[j].weight[k] += learning_rate * layer[i + 1].neuron[k].error * layer[i].neuron[j].output;
					}
				}
				for (int j = 0; j < layer[i + 1].neuron.Count; j ++)
				{
					layer[i].bias.output += learning_rate * layer[i + 1].neuron[j].error;
				}
			}
		}
		
		public void train(double[][] training_data_input, double[][] training_data_output, int epoch)
		{
			int current = 0;
			while (current < epoch)
			{
				for (int i = 0; i < training_data_input.GetLength(0); i ++)
				{
					Feed_Forward(training_data_input[i]);
					Back_Propagate(training_data_output[i]);
					Update_weights();
					
					// Console output
					string o = "";
					foreach (Neuron n in layer[layer.Count - 1].neuron)
					{
						if (n.output < 0.5) o += 0;
						else 				o += 1;
					}
					Console.WriteLine(o);
				}
				Console.WriteLine();
				current ++;
			}
		}
		public void compute(double[] input)
		{
			Feed_Forward(input);
			string o = "";
			foreach (Neuron n in layer[layer.Count - 1].neuron)
					{
						if (n.output < 0.5) o += 0;
						else 				o += 1;
					}
			Console.WriteLine(o);
		}
	}
	
	class Layer : Neural_Network
	{
		public List<Neuron> neuron = new List<Neuron>();
		public Neuron bias;
		
		public Layer(int size, int next_size)
		{
			for (int i = 0; i < size; i ++)
			{
				neuron.Add(new Neuron(next_size, false));				
			}
			bias = new Neuron(next_size, true);
		}
	}	
		
	class Neuron : Neural_Network
	{
		public List<double> weight = new List<double>();
		
		public Neuron(int next_size, bool isBias)
		{
			if (isBias) { output = 1; }
			else 
			{
				for (int i = 0; i < next_size; i ++)
				{
					weight.Add(rnd.NextDouble() * 2 - 1);
				}
			}
		}
		
		public double output;
		
		public double error;
	}
}
	

	
