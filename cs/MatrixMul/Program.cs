
using System.Diagnostics;
using System.Numerics.Tensors;

#pragma warning disable SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

Stopwatch sw = new();

// How long do we actually expect any of this to take?
// Each output requires n multiplications and n-1 additions.
// 8000 * 8000 * 8000 = 512,000,000,000 multiplications
// 8000 * 8000 * 7999 = 511,936,000,000
// So it's about 512 billion of each. 1024 billion operations
// In theory, with AVX2 my CPU (9900K, a Coffee Lake-S), can do 8 32-bit multiplications per cycle.
// So 32 billion cycles should be sufficient. At the base clock speed of 3.6GHz, that's about 9 seconds.
// But that's just on 1 core. I've got 8 cores, so if it were somehow possible to bring all cores to
// bear on the task, it should take about 1.1 seconds. That's not realistic, but it's a lower bound - if
// anything appears to go faster than that, it didn't do all the work!

// What if we go with a smaller problem size?
// 2048x2048 - 64x smaller, but also fits (exactly) into the 16MB L3 cache
// 2560 might be an interesting multiple. It's 25 MB, so large enough to blow the cache.
// It's a 3.125 linear reduction in size, so that's about 1/30th of the operations
// 512 is also an interesting threshold, because that's 2MB of data, which is the size of my L1 cache.
// 
// - 256x256?
// 2560 * 2560 * 2560 = 16,777,216,000 multiplications
// 2560 * 2560 * 2559 = 16,711,680 additions
// So it's about 16 million of each. 32 million operations
// In theory, with AVX2 my CPU (9900K, a Coffee Lake-S), can do 8 32-bit multiplications per cycle.
// So 4 million cycles should be sufficient. At the base clock speed of 3.6GHz, that's about 9 ms.
// But that's just on 1 core. I've got 8 cores, so if it were somehow possible to bring all cores to
// bear on the task, it should take about 1.1 seconds. That's not realistic, but it's a lower bound - if
// anything appears to go faster than that, it didn't do all the work!


TimeSpan openBlasRefTime = default;
TimeSpan naiveRefTime = default;
TimeSpan naiveFlippedRefTime = default;
TimeSpan simdFlippedRefTime = default;
TimeSpan simdFlippedParallelRefTime = default;
MulOps m256 = new(256);
MulOps m2000 = new(2000);
MulOps m2560 = new(2560);

// Warmup for tiered JIT purposes.
// In practice this doesn't seem to make a measureable difference. I suspect .NET 8.0's on-stack replacement
// optimizes early enough during the first call that it all runs more or less at full speed.
for (int i = 0; i < 20; i++)
{
    openBlasRefTime = m256.OpenBLASMatMul();
    naiveFlippedRefTime = m256.FlippedMatMul();
    naiveRefTime = m256.NaiveMatMul();
    simdFlippedRefTime = m256.FlippedMatMulSimd();
    simdFlippedParallelRefTime = m256.FlippedMatMulSimdParallelForLimited(divisor: 2);
    //m2560.NaiveMatMul();
}

Console.WriteLine();
Console.WriteLine("Warmed up.");
Console.WriteLine();

{
    {
        for (int i = 0; i < 10; ++i)
        {
            TimeSpan ob = m2000.OpenBLASMatMul();
        }
    }
    TimeSpan openBlas = m2000.OpenBLASMatMul();
    TimeSpan naive = m2000.NaiveMatMul();
    TimeSpan flipped = m2000.FlippedMatMul();
    TimeSpan simdFlipped = m2000.FlippedMatMulSimd();
    TimeSpan simdFlippedParallel16 = m2000.FlippedMatMulSimdParallelForLimited(divisor: 1);
    TimeSpan simdFlippedParallel8 = m2000.FlippedMatMulSimdParallelForLimited(divisor: 2);
    TimeSpan simdFlippedParallel4 = m2000.FlippedMatMulSimdParallelForLimited(divisor: 4);
    TimeSpan simdFlippedParallel2 = m2000.FlippedMatMulSimdParallelForLimited(divisor: 8);
    Console.WriteLine(
        $"Factors: size {2000.0 / 256.0}, calcs: {(((double)2000) * 2000 * 2000) / (256.0 * 256.0 * 256.0)} " +
        $"openBlas: {openBlas.TotalSeconds / openBlasRefTime.TotalSeconds}, " +
        $"flipped: {flipped.TotalSeconds / naiveFlippedRefTime.TotalSeconds}, " +
        $"flipped SIMD: {simdFlipped.TotalSeconds / simdFlippedRefTime.TotalSeconds}, " +
        $"flipped SIMD parallel (16): {simdFlippedParallel16.TotalSeconds / simdFlippedParallelRefTime.TotalSeconds}, " +
        $"flipped SIMD parallel (8): {simdFlippedParallel8.TotalSeconds / simdFlippedParallelRefTime.TotalSeconds}, " +
        $"flipped SIMD parallel (4): {simdFlippedParallel4.TotalSeconds / simdFlippedParallelRefTime.TotalSeconds}, " +
        $"flipped SIMD parallel (2): {simdFlippedParallel2.TotalSeconds / simdFlippedParallelRefTime.TotalSeconds}, " +
        $"naive: {naive.TotalSeconds / naiveRefTime.TotalSeconds}");
    Console.WriteLine($"naive:flipped: {naive.TotalSeconds / flipped.TotalSeconds}");
    Console.WriteLine($"flipped:flippedSimd: {flipped.TotalSeconds / simdFlipped.TotalSeconds}");
    Console.WriteLine($"flippedSimd:flippedSimdParallel 16 {simdFlipped.TotalSeconds / simdFlippedParallel16.TotalSeconds}");
    Console.WriteLine($"flippedSimd:flippedSimdParallel 8 {simdFlipped.TotalSeconds / simdFlippedParallel8.TotalSeconds}");
    Console.WriteLine($"flippedSimd:flippedSimdParallel 4 {simdFlipped.TotalSeconds / simdFlippedParallel4.TotalSeconds}");
    Console.WriteLine($"flippedSimd:flippedSimdParallel 2 {simdFlipped.TotalSeconds / simdFlippedParallel2.TotalSeconds}");

}

//for (int s = 256; s <= 1024; s += 256)
for (int s = 256; s <= 2560; s += 256)
{
    Console.WriteLine();
    MulOps m = new(s);
    TimeSpan openBlas = m.OpenBLASMatMul();
    TimeSpan naive = m.NaiveMatMul();
    TimeSpan flipped = m.FlippedMatMul();
    TimeSpan simdFlipped = m.FlippedMatMulSimd();
    TimeSpan simdFlippedParallel = m.FlippedMatMulSimdParallelForLimited(divisor: 2);
    Console.WriteLine(
        $"Factors: size {s / 256.0}, calcs: {(((double)s)*s*s) / (256.0*256.0*256.0)} " +
        $"flipped: {flipped.TotalSeconds / naiveFlippedRefTime.TotalSeconds}, " +
        $"flipped SIMD: {simdFlipped.TotalSeconds / simdFlippedRefTime.TotalSeconds}, " +
        $"flipped SIMD parallel: {simdFlippedParallel.TotalSeconds / simdFlippedParallelRefTime.TotalSeconds}, " +
        $"naive: {naive.TotalSeconds / naiveRefTime.TotalSeconds}");
    Console.WriteLine($"naive:flipped: {naive.TotalSeconds / flipped.TotalSeconds}");
    Console.WriteLine($"flipped:flippedSimd: {flipped.TotalSeconds / simdFlipped.TotalSeconds}");
    Console.WriteLine($"flippedSimd:flippedSimdParallel {simdFlipped.TotalSeconds / simdFlippedParallel.TotalSeconds}");
}

Console.WriteLine();
MulOps m8000 = new(8000);

//// Unconstrained parallelism.
//// 00:01:08.2
//// Note that when I had a bug in which I was using the column loop counter for the row loop, this took a lot
//// longer, which is an interesting insight into how locality of reference enables parallelism to do better here.
////  with wrong index for row 00:02:30.6
//m8000.FlippedMatMulSimdParallelFor();

//// What if we say exactly as many tasks as there are logical CPUs? Is that different from the default?
//// 1:06.4
//// I think that's too close to call different, with this experimental methodology
//m8000.FlippedMatMulSimdParallelForLimited(divisor: 1);

//// I've got 2 logical CPUs per actual core, so maybe 1 thread per core is optimal?
//// 0:21.8
////  with wrong index: 37.6s
//m8000.FlippedMatMulSimdParallelForLimited(divisor: 2);

//// I wondered if we were contending for memory bandwidth, meaning it'd be better to use even fewer
//// but apparently not:
//// 0:34.3
////  with wrong index: 1:08.1
//m8000.FlippedMatMulSimdParallelForLimited(divisor: 4);
//// 1:16.1
////  with wrong index 2:11.3
//m8000.FlippedMatMulSimdParallelForLimited(divisor: 8);

//// Very slow:

//// 2:09.0
////  with wrong index: 00:03:40.7
//m8000.FlippedMatMulSimd();

//// 10:11.3
//m8000.FlippedMatMul();

////const int n = 2560; // Should be about 30x faster.
//////const int n = 8000;
////sw.Start();
////float[] a1Values = new float[n * n];
////for (int i = 0; i < a1Values.Length; i++)
////{
////    a1Values[i] = Random.Shared.NextSingle() - 0.5f;
////}
////float[] a2Values = new float[n * n];
////for (int i = 0; i < a2Values.Length; i++)
////{
////    a2Values[i] = Random.Shared.NextSingle() - 0.5f;
////}
////sw.Stop();
////Console.WriteLine($"Rand: {sw.Elapsed}");

////float[] resultValues = new float[n * n];

////// With 8000x8000 this is too slow!
////// 2560x2560: 20.07s
////sw.Restart();
////for (int orow = 0; orow < n; orow++)
////{
////    //Console.WriteLine(orow);
////    for (int ocol = 0; ocol < n; ocol++)
////    {
////        float result = 0;
////        for (int i = 0; i < n; i++)
////        {
////            result += a1Values[orow * n + i] * a2Values[ocol * n + i];
////        }
////        resultValues[ocol + orow * n] = result;
////    }
////}
////sw.Stop();
////Console.WriteLine($"Naive for loop Matrix.Multiply, 2nd matrix flipped: {sw.Elapsed}");

////// With 8000x8000 this is too slow!
////// 2560x2560: 1:27.50
////sw.Restart();
////for (int orow = 0; orow < n; orow++)
////{
////    if ((orow & 0xff) == 0)
////    {
////        Console.WriteLine(orow); 
////    }
////    for (int ocol = 0; ocol < n; ocol++)
////    {
////        float result = 0;
////        for (int i = 0; i < n; i++)
////        {
////            result += a1Values[orow * n + i] * a2Values[i * n + ocol];
////        }
////        resultValues[ocol + orow * n] = result;
////    }
////}
////sw.Stop();
////Console.WriteLine($"Naive for loop Matrix.Multiply: {sw.Elapsed}");

////// 11 minutes 20 seconds
//////sw.Restart();
//////Parallel.For(0, n, orow =>
//////{
//////    Console.WriteLine(orow);
//////    for (int ocol = 0; ocol < n; ocol++)
//////    {
//////        float result = 0;
//////        for (int i = 0; i < n; i++)
//////        {
//////            result += a1Values[orow * n + i] * a2Values[i * n + ocol];
//////        }
//////        resultValues[ocol + orow * n] = result;
//////    }
//////});
//////sw.Stop();
//////Console.WriteLine($"Naive for loop Matrix.Multiply: {sw.Elapsed}");


////Matrix<float> ma1 = Matrix<float>.Build.DenseOfColumnMajor(n, n, a1Values);
////Matrix<float> ma2 = Matrix<float>.Build.DenseOfColumnMajor(n, n, a2Values);
////Matrix<float> mresult = Matrix<float>.Build.Dense(n, n);

////// 1:35
//////sw.Restart();
//////ma1.Multiply(ma2, mresult);
//////sw.Stop();
//////Console.WriteLine($"Math.Net Matrix.Multiply: {sw.Elapsed}");



////var ta1 = Tensor.Create(a1Values, [n, n]);
////var ta2 = Tensor.Create(a2Values, [n, n]);

////// 4:43
//////sw.Restart();
//////for (int orow = 0; orow < n; orow++)
//////{
//////    Console.WriteLine(orow);
//////    //var a2Col = ta2.Slice([0..1, 0..^1]);
//////    var a2Col = ta2[[0..1, 0..^1]];
//////    for (int ocol = 0; ocol < n; ocol++)
//////    {
//////        var a1Row = ta1[[orow..(orow + 1), 0..^1]];
//////        float result = Tensor.Dot(a1Row.AsReadOnlyTensorSpan(), a2Col.AsReadOnlyTensorSpan());
//////        resultValues[ocol + orow * n] = result;
//////    }
//////}
//////sw.Stop();
//////Console.WriteLine($"1 thread Tensor.Dot: {sw.Elapsed}");

////sw.Restart();
////Parallel.For(0, n, (int orow) =>
////{
////    Console.WriteLine(orow);
////    //var a2Col = ta2.Slice([0..1, 0..^1]);
////    var a2Col = ta2[[0..1, 0..^1]];
////    for (int ocol = 0; ocol < n; ocol++)
////    {
////        var a1Row = ta1[[orow..(orow + 1), 0..^1]];
////        float result = Tensor.Dot(a1Row.AsReadOnlyTensorSpan(), a2Col.AsReadOnlyTensorSpan());
////        resultValues[ocol + orow * n] = result;
////    }
////});
////sw.Stop();
////Console.WriteLine($"Parallel.For Tensor.Dot: {sw.Elapsed}");


////// This turns out to be pointless because Tensor.Multiply is element-wise multiplication,
////// and there is currently no support for matrix multiplication in System.Numerics.Tensors.
//////sw.Restart();
//////var ta3 = Tensor.Multiply(ta1.AsReadOnlyTensorSpan(), ta2.AsReadOnlyTensorSpan(), resultValues.AsTensorSpan([n, n]));
//////sw.Stop();
//////Console.WriteLine($"Tensor.Multiply: {sw.Elapsed}");
