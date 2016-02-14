//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#ifndef CUDASIFTD_H
#define CUDASIFTD_H

// 高斯金字塔有多层，由逐层降采样而得（512，256，128，64，32，16，8，4，2）
// 高斯金字塔每层图像有多组，每组对应不同高斯核，每组检测S（NUM_SCALES）个尺度的极值点
// DOG金字塔每组需S+2层图像（极值点由当前层周围8个和上下层各9个决定），
// 而DOG金字塔由高斯金字塔相邻两层相减得到，则高斯金字塔每组需S+3层图像，实际计算时S在3到5之间。
#define NUM_SCALES      5

// Scale down thread block width
// 降采样时，线程块对应原始图像区域的宽
#define SCALEDOWN_W   160

// Scale down thread block height
// 降采样时，线程块对应原始图像区域的高
#define SCALEDOWN_H    16

// Find point thread block width
#define MINMAX_W      126

// Find point thread block height
#define MINMAX_H        4

// Laplace thread block width
#define LAPLACE_W      56

// Number of laplace scales
#define LAPLACE_S   (NUM_SCALES+3)

// Laplace filter kernel radius
#define LAPLACE_R       4

//====================== Number of threads ====================//
// ScaleDown:               SCALEDOWN_W + 4
// LaplaceMulti:            (LAPLACE_W+2*LAPLACE_R)*LAPLACE_S
// FindPointsMulti:         MINMAX_W + 2
// ComputeOrientations:     128
// ExtractSiftDescriptors:  256

//====================== Number of blocks ====================//
// ScaleDown:               (width/SCALEDOWN_W) * (height/SCALEDOWN_H)
// LaplceMulti:             (width+2*LAPLACE_R)/LAPLACE_W * height
// FindPointsMulti:         (width/MINMAX_W)*NUM_SCALES * (height/MINMAX_H)
// ComputeOrientations:     numpts
// ExtractSiftDescriptors:  numpts

#endif
