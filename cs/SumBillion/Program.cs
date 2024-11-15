// See https://aka.ms/new-console-template for more information
using System.Diagnostics;

Stopwatch sw = new();

float[] numbers = new float[1000 * 1000 * 1000];
sw.Start();
for (int i = 0; i < numbers.Length; i++)
{
    numbers[i] = Random.Shared.NextSingle() - 0.5f;
}
sw.Stop();

Console.WriteLine($"Rand: {sw.Elapsed}");

sw.Restart();
float total = 0f;
for (int i = 0; i < numbers.Length; i++)
{
    total += numbers[i];
}
sw.Stop();

Console.WriteLine(total);
Console.WriteLine($"Sum (loop): {sw.Elapsed}");

sw.Restart();
total = System.Numerics.Tensors.TensorPrimitives.Sum(numbers);
sw.Stop();
Console.WriteLine(total);
Console.WriteLine($"Sum (TensorPrimitives): {sw.Elapsed}");
