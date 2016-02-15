//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#include <cudautils.h>
#include "cudaSiftD.h"
#include "cudaSift.h"

///////////////////////////////////////////////////////////////////////////////
// Kernel configuration
///////////////////////////////////////////////////////////////////////////////

__constant__ float d_Threshold[2];
__constant__ float d_Scales[8], d_Factor;
__constant__ float d_EdgeLimit;
__constant__ int d_MaxNumPoints;//SiftData分配的特征点数（maxPts）- 常量内存

__device__ unsigned int d_PointCounter[1];//SiftData中的可用特征点数（numPts）- 全局内存
__constant__ float d_Kernel1[5]; // 降采样时使用的1维高斯核 - 常量内存
__constant__ float d_Kernel2[12*16]; // 设备端的多尺度1维高斯核集合 - 常量内存

///////////////////////////////////////////////////////////////////////////////
// Lowpass filter an subsample image
///////////////////////////////////////////////////////////////////////////////
/**
 * 设备端的降采样
 * @param d_Result 降采样后的图像数据
 * @param d_Data   原始图像
 * @param width    原始图像的宽度
 * @param pitch    原始图像内存对齐宽度
 * @param height   原始图像的高度
 * @param newpitch 降采样后图像内存对齐宽度
 */
__global__ void ScaleDown(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
  __shared__ float inrow[SCALEDOWN_W + 4]; // 保存原始图像中的一行
  __shared__ float brow[5 * (SCALEDOWN_W / 2)]; // 保存按行高斯滤波后的结果，共5行
  __shared__ int yRead[SCALEDOWN_H + 4]; // 每个线程块对应原始图像的行索引的位置
  __shared__ int yWrite[SCALEDOWN_H + 4]; // 每个线程块对应降采样后的图像的行索引的位置
  #define dx2 (SCALEDOWN_W / 2)
  const int tx = threadIdx.x; // 当前线程在线程块中的列索引
  const int tx0 = tx + 0 * dx2; // 进行列滤波时的第一行索引
  const int tx1 = tx + 1 * dx2; // 进行列滤波时的第二行索引
  const int tx2 = tx + 2 * dx2; // 进行列滤波时的第三行索引
  const int tx3 = tx + 3 * dx2; // 进行列滤波时的第四行索引
  const int tx4 = tx + 4 * dx2; // 进行列滤波时的第五行索引
  const int xStart = blockIdx.x * SCALEDOWN_W; // 每个线程块对应原始图像的列索引的起始位置
  const int yStart = blockIdx.y * SCALEDOWN_H; // 每个线程块对应原始图像的行索引的起始位置
  const int xWrite = xStart / 2 + tx; // 保存数据的全局列索引
  const float *k = d_Kernel1; // 高斯核


  // 计算每个线程块对应的原始图像和降采样后的图像的行索引，并进行边界处理
  if (tx < SCALEDOWN_H + 4) {
    ///****** 是否有错误           ***********///
    ///****** y = yStart + tx - 2 **********///
    int y = yStart + tx - 1;
    y = (y < 0 ? 0 : y);
    y = (y >= height ? height - 1 : y);
    yRead[tx] = y * pitch; // 原始图像行索引
    yWrite[tx] = (yStart + tx - 4) / 2 * newpitch; // 降采样后图像行索引
  }
  __syncthreads();

  // 计算当前线程对应的列索引
  int xRead = xStart + tx - 2;
  xRead = (xRead < 0 ? 0 : xRead);
  xRead = (xRead >= width ? width - 1 : xRead);

  // 每5行（高斯核的大小）一起处理，直到处理完所有行
  for (int dy = 0; dy < SCALEDOWN_H + 4; dy += 5) {
    // 读原始图像的一行（5行中的第1行）
    inrow[tx] = d_Data[yRead[dy+0] + xRead];
    __syncthreads();

    // 进行行降采样及行高斯滤波（5行中的第1行）
    if (tx < dx2)
      brow[tx0] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
    __syncthreads();

    // 行滤波5行之后再进行列降采样和列高斯滤波（dy+0 = 10）
    // dy = 10, dy + 0 = 10，tx* (9, 5, 6, 7, 8)
    if (tx < dx2 && dy >= 4 && !(dy&1))
      d_Result[yWrite[dy+0] + xWrite] = k[2]*brow[tx2] + k[0]*(brow[tx0]+brow[tx4]) + k[1]*(brow[tx1]+brow[tx3]);

    if (dy < (SCALEDOWN_H + 3)) {
      // 读原始图像的一行（5行中的第2行）
      inrow[tx] = d_Data[yRead[dy+1] + xRead];
      __syncthreads();

      // 进行行降采样及行高斯滤波（5行中的第2行）
      if (tx < dx2)
      	brow[tx1] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();

      // 行滤波5行之后再进行列降采样和列高斯滤波
      // dy = 5, dy + 1 = 6，tx* (4, 5, 1, 2, 3)
      // dy = 15, dy + 1 = 16, tx* (14, 15, 11, 12, 13)
      if (tx < dx2 && dy >= 3 && (dy&1))
        d_Result[yWrite[dy+1] + xWrite] = k[2]*brow[tx3] + k[0]*(brow[tx1]+brow[tx0]) + k[1]*(brow[tx2]+brow[tx4]);
    }

    if (dy < (SCALEDOWN_H + 2)) {
      // 读原始图像的一行（5行中的第3行）
      inrow[tx] = d_Data[yRead[dy+2] + xRead];
      __syncthreads();

      // 进行行降采样及行高斯滤波（5行中的第3行）
      if (tx < dx2)
        brow[tx2] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();

      // 行滤波5行之后再进行列降采样和列高斯滤波
      // dy = 10, dy + 2 = 12，tx* (9, 10, 11, 7, 8)
      if (tx < dx2 && dy >= 2 && !(dy&1))
        d_Result[yWrite[dy+2] + xWrite] = k[2]*brow[tx4] + k[0]*(brow[tx2]+brow[tx1]) + k[1]*(brow[tx3]+brow[tx0]);
    }

    if (dy < (SCALEDOWN_H + 1)) {
      // 读原始图像的一行（5行中的第4行）
      inrow[tx] = d_Data[yRead[dy+3] + xRead];
      __syncthreads();

      // 进行行降采样及行高斯滤波（5行中的第4行）
      if (tx < dx2)
        brow[tx3] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();

      // 行滤波5行之后再进行列降采样和列高斯滤波
      // dy = 5, dy + 3 = 8，tx* (4, 5, 6, 7, 3)
      // dy = 15, dy + 3 = 18, tx* (14, 15, 16, 17, 13)
      if (tx < dx2 && dy >= 1 && (dy&1))
        d_Result[yWrite[dy+3] + xWrite] = k[2]*brow[tx0] + k[0]*(brow[tx3]+brow[tx2]) + k[1]*(brow[tx4]+brow[tx1]);
    }

    if (dy < SCALEDOWN_H) {
      // 读原始图像的一行（5行中的第5行）
      inrow[tx] = d_Data[yRead[dy+4] + xRead];
      __syncthreads();

      // 进行行降采样及行高斯滤波（5行中的第5行）
      if (tx<dx2)
        brow[tx4] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();

      // 行滤波5行之后再进行列降采样和列高斯滤波
      // dy = 0, dy + 4 = 4，tx* (0, 0, 1, 2, 3)
      // dy = 10, dy + 4 = 14, tx* (9, 10, 11, 12, 13)
      if (tx<dx2 && !(dy&1))
        d_Result[yWrite[dy+4] + xWrite] = k[2]*brow[tx1] + k[0]*(brow[tx4]+brow[tx3]) + k[1]*(brow[tx0]+brow[tx2]);
    }

    __syncthreads();
  }
}


__global__ void ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftPoint *d_sift, int fstPts, float subsampling)
{
  __shared__ float gauss[16];
  __shared__ float buffer[128];
  __shared__ float sums[128];

  const int tx = threadIdx.x; // 0 -> 16
  const int ty = threadIdx.y; // 0 -> 16
  const int idx = ty*16 + tx;
  const int bx = blockIdx.x + fstPts;  // 0 -> numPts
  if (ty==0)
    gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);
  buffer[idx] = 0.0f;
  __syncthreads();

  // Compute angles and gradients
  float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
  float sina = sinf(theta);           // cosa -sina
  float cosa = cosf(theta);           // sina  cosa
  float scale = 12.0f/16.0f*d_sift[bx].scale;
  float ssina = scale*sina;
  float scosa = scale*cosa;

  for (int y=ty;y<16;y+=8) {
    float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa - (y-7.5f)*ssina;
    float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina + (y-7.5f)*scosa;
    float dx = tex2D<float>(texObj, xpos+cosa, ypos+sina) -
      tex2D<float>(texObj, xpos-cosa, ypos-sina);
    float dy = tex2D<float>(texObj, xpos-sina, ypos+cosa) -
      tex2D<float>(texObj, xpos+sina, ypos-cosa);
    float grad = gauss[y]*gauss[tx] * sqrtf(dx*dx + dy*dy);
    float angf = 4.0f/3.1415f*atan2f(dy, dx) + 4.0f;

    int hori = (tx + 2)/4 - 1;      // Convert from (tx,y,angle) to bins
    float horf = (tx - 1.5f)/4.0f - hori;
    float ihorf = 1.0f - horf;
    int veri = (y + 2)/4 - 1;
    float verf = (y - 1.5f)/4.0f - veri;
    float iverf = 1.0f - verf;
    int angi = angf;
    int angp = (angi<7 ? angi+1 : 0);
    angf -= angi;
    float iangf = 1.0f - angf;

    int hist = 8*(4*veri + hori);   // Each gradient measure is interpolated
    int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
    int p2 = angp + hist;
    if (tx>=2) {
      float grad1 = ihorf*grad;
      if (y>=2) {   // Upper left
        float grad2 = iverf*grad1;
	atomicAdd(buffer + p1, iangf*grad2);
	atomicAdd(buffer + p2,  angf*grad2);
      }
      if (y<=13) {  // Lower left
        float grad2 = verf*grad1;
	atomicAdd(buffer + p1+32, iangf*grad2);
	atomicAdd(buffer + p2+32,  angf*grad2);
      }
    }
    if (tx<=14) {
      float grad1 = horf*grad;
      if (y>=2) {    // Upper right
        float grad2 = iverf*grad1;
	atomicAdd(buffer + p1+8, iangf*grad2);
	atomicAdd(buffer + p2+8,  angf*grad2);
      }
      if (y<=13) {   // Lower right
        float grad2 = verf*grad1;
	atomicAdd(buffer + p1+40, iangf*grad2);
	atomicAdd(buffer + p2+40,  angf*grad2);
      }
    }
  }
  __syncthreads();

  // Normalize twice and suppress peaks first time
  if (idx<64)
    sums[idx] = buffer[idx]*buffer[idx] + buffer[idx+64]*buffer[idx+64];
  __syncthreads();
  if (idx<32) sums[idx] = sums[idx] + sums[idx+32];
  __syncthreads();
  if (idx<16) sums[idx] = sums[idx] + sums[idx+16];
  __syncthreads();
  if (idx<8)  sums[idx] = sums[idx] + sums[idx+8];
  __syncthreads();
  if (idx<4)  sums[idx] = sums[idx] + sums[idx+4];
  __syncthreads();
  float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
  buffer[idx] = buffer[idx] * rsqrtf(tsum1);

  if (buffer[idx]>0.2f)
    buffer[idx] = 0.2f;
  __syncthreads();
  if (idx<64)
    sums[idx] = buffer[idx]*buffer[idx] + buffer[idx+64]*buffer[idx+64];
  __syncthreads();
  if (idx<32) sums[idx] = sums[idx] + sums[idx+32];
  __syncthreads();
  if (idx<16) sums[idx] = sums[idx] + sums[idx+16];
  __syncthreads();
  if (idx<8)  sums[idx] = sums[idx] + sums[idx+8];
  __syncthreads();
  if (idx<4)  sums[idx] = sums[idx] + sums[idx+4];
  __syncthreads();
  float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];

  float *desc = d_sift[bx].data;
  desc[idx] = buffer[idx] * rsqrtf(tsum2);
  if (idx==0) {
    d_sift[bx].xpos *= subsampling;
    d_sift[bx].ypos *= subsampling;
    d_sift[bx].scale *= subsampling;
  }
}


__global__ void ComputeOrientations(cudaTextureObject_t texObj, SiftPoint *d_Sift, int fstPts)
{
  __shared__ float hist[64];
  __shared__ float gauss[11];
  const int tx = threadIdx.x;
  const int bx = blockIdx.x + fstPts;
  float i2sigma2 = -1.0f/(4.5f*d_Sift[bx].scale*d_Sift[bx].scale);
  if (tx<11)
    gauss[tx] = exp(i2sigma2*(tx-5)*(tx-5));
  if (tx<64)
    hist[tx] = 0.0f;
  __syncthreads();
  float xp = d_Sift[bx].xpos - 5.0f;
  float yp = d_Sift[bx].ypos - 5.0f;
  int yd = tx/11;
  int xd = tx - yd*11;
  float xf = xp + xd;
  float yf = yp + yd;
  if (yd<11) {
    float dx = tex2D<float>(texObj, xf+1.0, yf) - tex2D<float>(texObj, xf-1.0, yf);
    float dy = tex2D<float>(texObj, xf, yf+1.0) - tex2D<float>(texObj, xf, yf-1.0);
    int bin = 16.0f*atan2f(dy, dx)/3.1416f + 16.5f;
    if (bin>31)
      bin = 0;
    float grad = sqrtf(dx*dx + dy*dy);
    atomicAdd(&hist[bin], grad*gauss[xd]*gauss[yd]);
  }
  __syncthreads();
  int x1m = (tx>=1 ? tx-1 : tx+31);
  int x1p = (tx<=30 ? tx+1 : tx-31);
  if (tx<32) {
    int x2m = (tx>=2 ? tx-2 : tx+30);
    int x2p = (tx<=29 ? tx+2 : tx-30);
    hist[tx+32] = 6.0f*hist[tx] + 4.0f*(hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
  }
  __syncthreads();
  if (tx<32) {
    float v = hist[32+tx];
    hist[tx] = (v>hist[32+x1m] && v>=hist[32+x1p] ? v : 0.0f);
  }
  __syncthreads();
  if (tx==0) {
    float maxval1 = 0.0;
    float maxval2 = 0.0;
    int i1 = -1;
    int i2 = -1;
    for (int i=0;i<32;i++) {
      float v = hist[i];
      if (v>maxval1) {
	maxval2 = maxval1;
	maxval1 = v;
	i2 = i1;
	i1 = i;
      } else if (v>maxval2) {
	maxval2 = v;
	i2 = i;
      }
    }
    float val1 = hist[32+((i1+1)&31)];
    float val2 = hist[32+((i1+31)&31)];
    float peak = i1 + 0.5f*(val1-val2) / (2.0f*maxval1-val1-val2);
    d_Sift[bx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);
    if (maxval2>0.8f*maxval1 && false) {
      float val1 = hist[32+((i2+1)&31)];
      float val2 = hist[32+((i2+31)&31)];
      float peak = i2 + 0.5f*(val1-val2) / (2.0f*maxval2-val1-val2);
      unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
      if (idx<d_MaxNumPoints) {
	d_Sift[idx].xpos = d_Sift[bx].xpos;
	d_Sift[idx].ypos = d_Sift[bx].ypos;
	d_Sift[idx].scale = d_Sift[bx].scale;
	d_Sift[idx].sharpness = d_Sift[bx].sharpness;
	d_Sift[idx].edgeness = d_Sift[bx].edgeness;
	d_Sift[idx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);;
	d_Sift[idx].subsampling = d_Sift[bx].subsampling;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Subtract two images (multi-scale version)
///////////////////////////////////////////////////////////////////////////////

 __global__ void FindPointsMulti(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, int nScales, float subsampling)
{
  #define MEMWID (MINMAX_W + 2)  // (128)
  __shared__ float ymin1[MEMWID], ymin2[MEMWID], ymin3[MEMWID];
  __shared__ float ymax1[MEMWID], ymax2[MEMWID], ymax3[MEMWID];
  __shared__ unsigned int cnt;
  __shared__ unsigned short points[96];

  int tx = threadIdx.x; // 当前线程的列索引
  int block = blockIdx.x / nScales; // 当前线程对应的 nScales，中间组数
  int scale = blockIdx.x - nScales * block; // 当前线程对应某个中间组的列数
  int minx = block * MINMAX_W;
  int maxx = min(minx + MINMAX_W, width);
  int xpos = minx + tx;
  int size = pitch * height;
  int ptr = size * scale + max(min(xpos-1, width-1), 0);

  if (tx == 0)
    cnt = 0;
  __syncthreads();

  int yloops = min(height - MINMAX_H * blockIdx.y, MINMAX_H);
  for (int y=0; y < yloops; y++) {

    int ypos = MINMAX_H * blockIdx.y + y;
    int yptr0 = ptr + max(0, ypos-1) * pitch;
    int yptr1 = ptr + ypos * pitch;
    int yptr2 = ptr + min(height-1, ypos+1) * pitch;

    {
      float d10 = d_Data0[yptr0];
      float d11 = d_Data0[yptr1];
      float d12 = d_Data0[yptr2];
      ymin1[tx] = fminf(fminf(d10, d11), d12);
      ymax1[tx] = fmaxf(fmaxf(d10, d11), d12);
    }

    {
      float d30 = d_Data0[yptr0 + 2*size];
      float d31 = d_Data0[yptr1 + 2*size];
      float d32 = d_Data0[yptr2 + 2*size];
      ymin3[tx] = fminf(fminf(d30, d31), d32);
      ymax3[tx] = fmaxf(fmaxf(d30, d31), d32);
    }

    float d20 = d_Data0[yptr0 + 1*size];
    float d21 = d_Data0[yptr1 + 1*size];
    float d22 = d_Data0[yptr2 + 1*size];
    ymin2[tx] = fminf(fminf(ymin1[tx], fminf(fminf(d20, d21), d22)), ymin3[tx]);
    ymax2[tx] = fmaxf(fmaxf(ymax1[tx], fmaxf(fmaxf(d20, d21), d22)), ymax3[tx]);

    __syncthreads();

    if (tx > 0 && tx < MINMAX_W + 1 && xpos <= maxx) {
      if (d21 < d_Threshold[1]) {
        float minv = fminf(fminf(fminf(ymin2[tx-1], ymin2[tx+1]), ymin1[tx]), ymin3[tx]);
        minv = fminf(fminf(minv, d20), d22);
        if (d21<minv) {
          int pos = atomicInc(&cnt, 31);
          points[3*pos+0] = xpos - 1;
          points[3*pos+1] = ypos;
          points[3*pos+2] = scale;
        }
      }

      if (d21 > d_Threshold[0]) {
        float maxv = fmaxf(fmaxf(fmaxf(ymax2[tx-1], ymax2[tx+1]), ymax1[tx]), ymax3[tx]);
        maxv = fmaxf(fmaxf(maxv, d20), d22);
        if (d21>maxv) {
          int pos = atomicInc(&cnt, 31);
          points[3*pos+0] = xpos - 1;
          points[3*pos+1] = ypos;
          points[3*pos+2] = scale;
        }
      }

    }

    __syncthreads();
  }

  if (tx < cnt) {
    int xpos = points[3*tx+0];
    int ypos = points[3*tx+1];
    int scale = points[3*tx+2];
    int ptr = xpos + (ypos + (scale+1)*height)*pitch;
    float val = d_Data0[ptr];
    float *data1 = &d_Data0[ptr];
    float dxx = 2.0f*val - data1[-1] - data1[1];
    float dyy = 2.0f*val - data1[-pitch] - data1[pitch];
    float dxy = 0.25f*(data1[+pitch+1] + data1[-pitch-1] - data1[-pitch+1] - data1[+pitch-1]);
    float tra = dxx + dyy;
    float det = dxx*dyy - dxy*dxy;

    if (tra*tra<d_EdgeLimit*det) {
      float edge = __fdividef(tra*tra, det);
      float dx = 0.5f*(data1[1] - data1[-1]);
      float dy = 0.5f*(data1[pitch] - data1[-pitch]);
      float *data0 = d_Data0 + ptr - height*pitch;
      float *data2 = d_Data0 + ptr + height*pitch;
      float ds = 0.5f*(data0[0] - data2[0]);
      float dss = 2.0f*val - data2[0] - data0[0];
      float dxs = 0.25f*(data2[1] + data0[-1] - data0[1] - data2[-1]);
      float dys = 0.25f*(data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
      float idxx = dyy*dss - dys*dys;
      float idxy = dys*dxs - dxy*dss;
      float idxs = dxy*dys - dyy*dxs;
      float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
      float idyy = dxx*dss - dxs*dxs;
      float idys = dxy*dxs - dxx*dys;
      float idss = dxx*dyy - dxy*dxy;
      float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
      float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
      float pds = idet*(idxs*dx + idys*dy + idss*ds);

      if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
        pdx = __fdividef(dx, dxx);
      	pdy = __fdividef(dy, dyy);
      	pds = __fdividef(ds, dss);
      }

      float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
      int maxPts = d_MaxNumPoints;
      unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
      idx = (idx>=maxPts ? maxPts-1 : idx);
      d_Sift[idx].xpos = xpos + pdx;
      d_Sift[idx].ypos = ypos + pdy;
      d_Sift[idx].scale = d_Scales[scale] * exp2f(pds*d_Factor);
      d_Sift[idx].sharpness = val + dval;
      d_Sift[idx].edgeness = edge;
      d_Sift[idx].subsampling = subsampling;
    }
  }
}

/**
 * 计算一层高斯金字塔和DOG金字塔
 * @param texObj   图像的纹理数据
 * @param d_Result DOG金字塔的某层
 * @param width    图像的宽度
 * @param pitch    内存对齐宽度
 * @param height   图像高度
 */
 __global__ void LaplaceMulti(cudaTextureObject_t texObj, float *d_Result, int width, int pitch, int height)
{
  __shared__ float data1[(LAPLACE_W + 2 * LAPLACE_R) * LAPLACE_S];
  __shared__ float data2[LAPLACE_W * LAPLACE_S];
  const int tx = threadIdx.x; // 当前线程的列索引
  const int xp = blockIdx.x*LAPLACE_W + tx; // 当前组的列索引
  const int yp = blockIdx.y; // 每组的第几行
  const int scale = threadIdx.y; // 当前层的第几组
  float *kernel = d_Kernel2 + scale*16; // 每行线程对应组高斯核
  float *sdata1 = data1 + (LAPLACE_W + 2 * LAPLACE_R) * scale; // 每行线程对应的共享内存
  float x = xp - 3.5;
  float y = yp + 0.5;

  // 列高斯滤波
  sdata1[tx] = kernel[4] * tex2D<float>(texObj, x, y) +
    kernel[3] * (tex2D<float>(texObj, x, y-1.0) + tex2D<float>(texObj, x, y+1.0)) +
    kernel[2] * (tex2D<float>(texObj, x, y-2.0) + tex2D<float>(texObj, x, y+2.0)) +
    kernel[1] * (tex2D<float>(texObj, x, y-3.0) + tex2D<float>(texObj, x, y+3.0)) +
    kernel[0] * (tex2D<float>(texObj, x, y-4.0) + tex2D<float>(texObj, x, y+4.0));
  __syncthreads();

  float *sdata2 = data2 + LAPLACE_W * scale;

  // 行高斯滤波
  if (tx < LAPLACE_W) {
    sdata2[tx] = kernel[4]*sdata1[tx+4] +
      kernel[3] * (sdata1[tx+3] + sdata1[tx+5]) +
      kernel[2] * (sdata1[tx+2] + sdata1[tx+6]) +
      kernel[1] * (sdata1[tx+1] + sdata1[tx+7]) +
      kernel[0] * (sdata1[tx+0] + sdata1[tx+8]);
  }
  __syncthreads();

  // 差分
  if (tx < LAPLACE_W && scale < LAPLACE_S-1 && xp<width)
    d_Result[scale * height * pitch + yp * pitch + xp] = sdata2[tx] - sdata2[tx + LAPLACE_W];
}


