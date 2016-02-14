#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cstdio>
#include <iostream>

#ifdef WIN32
#include <intrin.h>
#endif

#define safeCall(err)       __safeCall(err, __FILE__, __LINE__)
#define safeThreadSync()    __safeThreadSync(__FILE__, __LINE__)
#define checkMsg(msg)       __checkMsg(msg, __FILE__, __LINE__)

/**
 * 检测CUDA函数是否正常运行
 * @param err  错误信息
 * @param file 调用函数的文件
 * @param line 调用函数的位置
 */
inline void __safeCall(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err) {
    fprintf(stderr, "safeCall() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

/**
 * 检测线程同步函数是否正常
 * @param file 调用同步函数的文件
 * @param line 调用同步函数的位置
 */
inline void __safeThreadSync(const char *file, const int line)
{
  cudaError err = cudaThreadSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "threadSynchronize() Driver API error in file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

/**
 * 检测最近的错误信息
 * @param errorMessage 错误信息
 * @param file         调用同步函数的文件
 * @param line         调用同步函数的位置
 */
inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

/**
 * 设备初始化
 * @param  dev 指定调用的设备号
 * @return
 */
inline bool deviceInit(int dev)
{
  // 获取设备个数
  int deviceCount;
  safeCall(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
    return false;
  }

  if (dev < 0) dev = 0; // 如果输入dev小于0，默认调用设备0
  if (dev > deviceCount-1) dev = deviceCount - 1; // 如果输入dev大于设备个数，默认调用最后一个设备

  // 检测设备的计算能力，需要不小于1.0
  cudaDeviceProp deviceProp;
  safeCall(cudaGetDeviceProperties(&deviceProp, dev));
  if (deviceProp.major < 1) {
    fprintf(stderr, "error: device does not support CUDA.\n");
    return false;
  }
  safeCall(cudaSetDevice(dev));
  return true;
}

/**
 * 设备端计时，使用 cudaEvent_t
 */
class TimerGPU {
public:
  cudaEvent_t start, stop;
  cudaStream_t stream;

  // 开始计时
  TimerGPU(cudaStream_t stream_ = 0) : stream(stream_) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
  }
  ~TimerGPU() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // 结束计时
  float read() {
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    return time;
  }
};

/**
 * 主机端计时
 */
class TimerCPU
{
  static const int bits = 10;
public:
  long long beg_clock;
  float freq;

  // 开始计时
  TimerCPU(float freq_) : freq(freq_) {   // freq = clock frequency in MHz
    beg_clock = getTSC(bits);
  }
  long long getTSC(int bits) {
#ifdef WIN32
    return __rdtsc()/(1LL<<bits);
#else
    unsigned int low, high;
    __asm__(".byte 0x0f, 0x31" :"=a" (low), "=d" (high));
    return ((long long)high<<(32-bits)) | ((long long)low>>bits);
#endif
  }

  // 结束计时
  float read() {
    long long end_clock = getTSC(bits);
    long long Kcycles = end_clock - beg_clock;
    float time = (float)(1<<bits)*Kcycles/freq/1e3f;
    return time;
  }
};


#endif

