//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

class CudaImage {
public:
  int width, height;
  int pitch;
  float *h_data;
  float *d_data;
  float *t_data;
  bool d_internalAlloc;
  bool h_internalAlloc;
public:
  CudaImage();
  ~CudaImage();
  void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL);
  double Download();
  double Readback();
  double InitTexture();
  double CopyToTexture(CudaImage &dst, bool host);
};

int iDivUp(int a, int b); // 除：向上取整
int iDivDown(int a, int b); // 除：向下取整
int iAlignUp(int a, int b); // 使a是b整数倍：a向上扩充
int iAlignDown(int a, int b); // 使a是b的整数倍：a向下裁剪
void StartTimer(unsigned int *hTimer);
double StopTimer(unsigned int hTimer);

#endif // CUDAIMAGE_H
