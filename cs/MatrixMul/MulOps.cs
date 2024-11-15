using OpenBlasSharp;

using System.Diagnostics;
using System.Numerics.Tensors;

internal class MulOps
{
    private static Stopwatch sw = new();

    private readonly int n;
    private readonly float[] a1Values;
    private readonly float[] a2Values;
    private readonly float[] resultValues;

    public MulOps(int n)
    {
        this.n = n;
        a1Values = new float[n * n];
        a2Values = new float[n * n];

        resultValues = new float[n * n];
        sw.Start();
        for (int i = 0; i < a1Values.Length; i++)
        {
            a1Values[i] = Random.Shared.NextSingle() - 0.5f;
        }
        for (int i = 0; i < a2Values.Length; i++)
        {
            a2Values[i] = Random.Shared.NextSingle() - 0.5f;
        }

        sw.Stop();
        //Console.WriteLine($"Rand: {sw.Elapsed}");
    }

    public unsafe TimeSpan OpenBLASMatMul()
    {
        sw.Restart();
        fixed (float* pa = a1Values)
        fixed (float* pb = a2Values)
        fixed (float* pc = resultValues)
        {
            // Calculate c = a * b.
            Blas.Sgemm(
                Order.ColMajor,
                Transpose.NoTrans,
                Transpose.NoTrans,
                n, n, n,
                1.0f,
                pa, n,
                pb, n,
                0.0f,
                pc, n);
        }
        sw.Stop();
        Console.WriteLine($"{n}x{n} OpenBLAS for loop Matrix.Multiply: {sw.Elapsed}");
        return sw.Elapsed;
    }

    public TimeSpan NaiveMatMul()
    {
        sw.Restart();

        for (int orow = 0; orow < n; orow++)
        {
            for (int ocol = 0; ocol < n; ocol++)
            {
                float result = 0;
                for (int i = 0; i < n; i++)
                {
                    result += a1Values[orow * n + i] * a2Values[i * n + ocol];
                }
                resultValues[ocol + orow * n] = result;
            }
        }

        sw.Stop();
        Console.WriteLine($"{n}x{n} Naive for loop Matrix.Multiply: {sw.Elapsed}");
        return sw.Elapsed;
    }

    public TimeSpan FlippedMatMul()
    {
        sw.Restart();
        for (int orow = 0; orow < n; orow++)
        {
            //Console.WriteLine(orow);
            for (int ocol = 0; ocol < n; ocol++)
            {
                float result = 0;
                for (int i = 0; i < n; i++)
                {
                    result += a1Values[orow * n + i] * a2Values[ocol * n + i];
                }
                resultValues[ocol + orow * n] = result;
            }
        }
        sw.Stop();
        Console.WriteLine($"{n}x{n} Naive for loop Matrix.Multiply, 2nd matrix flipped: {sw.Elapsed}");
        return sw.Elapsed;
    }

    public TimeSpan FlippedMatMulSimd()
    {
        sw.Restart();
        ReadOnlySpan<float> a1Data = a1Values.AsSpan();
        ReadOnlySpan<float> a2Data = a2Values.AsSpan();
        for (int orow = 0; orow < n; orow++)
        {
            ReadOnlySpan<float> a1Row = a1Data.Slice(orow * n, n);
            //Console.WriteLine(orow);
            for (int ocol = 0; ocol < n; ocol++)
            {
                ReadOnlySpan<float> a2Column = a2Data.Slice(ocol * n, n);
                float result = TensorPrimitives.Dot(a1Row, a2Column);
                resultValues[ocol + orow * n] = result;
            }
        }
        sw.Stop();
        Console.WriteLine($"{n}x{n} for loop with inner SIMD Matrix.Multiply, 2nd matrix flipped: {sw.Elapsed}");
        return sw.Elapsed;
    }

    public TimeSpan FlippedMatMulSimdParallelFor()
    {
        sw.Restart();
        Parallel.For(0, n, orow =>
        {
            ReadOnlySpan<float> a1Data = a1Values.AsSpan();
            ReadOnlySpan<float> a2Data = a2Values.AsSpan();
            ReadOnlySpan<float> a1Row = a1Data.Slice(orow * n, n);
            //Console.WriteLine(orow);
            for (int ocol = 0; ocol < n; ocol++)
            {
                ReadOnlySpan<float> a2Column = a2Data.Slice(ocol * n, n);
                float result = TensorPrimitives.Dot(a1Row, a2Column);
                resultValues[ocol + orow * n] = result;
            }
        });
        sw.Stop();
        Console.WriteLine($"{n}x{n} Parallel.For with inner SIMD Matrix.Multiply, 2nd matrix flipped: {sw.Elapsed}");
        return sw.Elapsed;
    }

    public TimeSpan FlippedMatMulSimdParallelForLimited(int divisor)
    {
        sw.Restart();
        Parallel.For(
            0,
            n,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount / divisor },
            orow =>
            {
                ReadOnlySpan<float> a1Data = a1Values.AsSpan();
                ReadOnlySpan<float> a2Data = a2Values.AsSpan();
                //Console.WriteLine(orow);
                ReadOnlySpan<float> a1Row = a1Data.Slice(orow * n, n);
                for (int ocol = 0; ocol < n; ocol++)
                {
                    ReadOnlySpan<float> a2Column = a2Data.Slice(ocol * n, n);
                    float result = TensorPrimitives.Dot(a1Row, a2Column);
                    resultValues[ocol + orow * n] = result;
                }
            });
        sw.Stop();
        Console.WriteLine($"{n}x{n} Parallel.For (max {Environment.ProcessorCount / divisor}) with inner SIMD Matrix.Multiply, 2nd matrix flipped: {sw.Elapsed}");
        return sw.Elapsed;
    }

    // Next: can we pin specific threads to specific cores? With Parallel.For we seem to smoosh the load across all the
    // cores when trying to use a subset. That seems likely to make the cache less effective. So a targeted approach in
    // which we 
}