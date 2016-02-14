//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#include <cstdio>

#include "cudautils.h"
#include "cudaImage.h"

/**
 * 除：向上取整
 * @param  a 被除数
 * @param  b 除数
 * @return   商
 */
int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }

/**
 * 除：向下取整
 * @param  a 被除数
 * @param  b 除数
 * @return   商
 */
int iDivDown(int a, int b) { return a/b; }

/**
 * 使a是b整数倍:a向上扩充
 * @param  a
 * @param  b
 * @return
 */
int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }

/**
 * 使a是b整数倍:a向下裁剪
 * @param  a
 * @param  b
 * @return
 */
int iAlignDown(int a, int b) { return a - a%b; }

/**
 * 为图像数据分配主机端和设备端内存
 * @param w       图像宽
 * @param h       图像高
 * @param p       图像内存对齐参数
 * @param host    图像是否存储于主机端
 * @param devmem  设备端数据头指针
 * @param hostmem 主机端数据头指针
 */
void CudaImage::Allocate(int w, int h, int p, bool host, float *devmem, float *hostmem)
{
  width = w;
  height = h;
  pitch = p;
  d_data = devmem;
  h_data = hostmem;
  t_data = NULL;
  if (devmem==NULL) {
    safeCall(cudaMallocPitch((void **)&d_data, (size_t*)&pitch, (size_t)(sizeof(float)*width), (size_t)height));
    pitch /= sizeof(float);
    if (d_data==NULL)
      printf("Failed to allocate device data\n");
    d_internalAlloc = true;
  }
  if (host && hostmem==NULL) {
    h_data = (float *)malloc(sizeof(float)*pitch*height);
    h_internalAlloc = true;
  }
}

CudaImage::CudaImage() :
  width(0), height(0), d_data(NULL), h_data(NULL), t_data(NULL), d_internalAlloc(false), h_internalAlloc(false)
{

}

CudaImage::~CudaImage()
{
  if (d_internalAlloc && d_data!=NULL)
    safeCall(cudaFree(d_data));
  d_data = NULL;
  if (h_internalAlloc && h_data!=NULL)
    free(h_data);
  h_data = NULL;
  if (t_data!=NULL)
    safeCall(cudaFreeArray((cudaArray *)t_data));
  t_data = NULL;
}

/**
 * 数据从主机端拷贝到设备端
 * @return 数据拷贝时间
 */
double CudaImage::Download()
{
  TimerGPU timer(0);
  int p = sizeof(float)*pitch;
  if (d_data!=NULL && h_data!=NULL)
    safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Download time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

/**
 * 数据从设备端拷到主机端
 * @return 数据拷贝时间
 */
double CudaImage::Readback()
{
  TimerGPU timer(0);
  int p = sizeof(float)*pitch;
  safeCall(cudaMemcpy2D(h_data, sizeof(float)*width, d_data, p, sizeof(float)*width, height, cudaMemcpyDeviceToHost));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Readback time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

/**
 * 纹理初始化
 * @return 纹理初始化时间
 */
double CudaImage::InitTexture()
{
  TimerGPU timer(0);
  // cudaMallocArray() 需要使用 cudaCreateChannelDesc() 创建的格式描述。
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>();
  safeCall(cudaMallocArray((cudaArray **)&t_data, &t_desc, pitch, height));
  if (t_data==NULL)
    printf("Failed to allocated texture data\n");
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("InitTexture time =            %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

/**
 * 拷贝数据到纹理内存
 * @param  dst 待拷贝的图像数据
 * @param  host dst的数据是否位于主机内存中（与数据拷贝的方向有关）
 * @return      数据拷贝时间
 */
double CudaImage::CopyToTexture(CudaImage &dst, bool host)
{
  if (dst.t_data==NULL) {
    printf("Error CopyToTexture: No texture data\n");
    return 0.0;
  }
  if ((!host || h_data==NULL) && (host || d_data==NULL)) {
    printf("Error CopyToTexture: No source data\n");
    return 0.0;
  }
  TimerGPU timer(0);
  if (host)
    safeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, h_data, sizeof(float)*pitch*dst.height, cudaMemcpyHostToDevice));
  else
    safeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, d_data, sizeof(float)*pitch*dst.height, cudaMemcpyDeviceToDevice));
  safeCall(cudaThreadSynchronize());
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("CopyToTexture time =          %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}
