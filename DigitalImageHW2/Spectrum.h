#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// 避免重复引入stb_image, 只在main.cpp中引入一次即可.

#include<algorithm>
#include<memory>
#include<complex>
#include<functional>

#include"Image.h"

typedef std::complex<double> ComplexDouble;
// 正变换函数指针, W, H, M, N, 两个引用是输出.
typedef std::function<std::shared_ptr<ComplexDouble[]>(std::shared_ptr<ComplexDouble[]>, int, int, int&, int&)> SpectrumTransformFunc;  
// 逆变换函数指针, 第一个int是width, 第二个int是height
typedef std::function<std::shared_ptr<ComplexDouble[]>(std::shared_ptr<ComplexDouble[]>, int, int)> SpectrumInverseTransformFunc;
typedef std::function<ComplexDouble(ComplexDouble, int, int, int, int)> SpectrumFilterFunc;  // 滤波器函数指针,   第一个int是横轴的u，第二个int是纵轴的v, 第三个int是横轴中心，第四个int是纵轴中心

class Spectrum {
public:
	int M, N, H, W, shifted;   // M 对应x，是横轴；N 对应y，是纵轴.  H, W 是原图的宽高
	std::shared_ptr<ComplexDouble[]> rawdata;

	// 将图片转化成complex，处理中心化的问题，传入func进行变换.
	Spectrum(Image img, SpectrumTransformFunc transform_func, int shift);
	// 将频谱传入func进行逆变换，处理中心化问题, 变成u8 Image.
	Image InverseTransform(SpectrumInverseTransformFunc inversefunc);
	void ApplyFilter(SpectrumFilterFunc filterfunc);         // 直接在本对象上发生修改.
	void LowPassFilter(int thresh);
	void BandPassFilter(int l_thresh, int h_thresh);
	void HighPassFilter(int thresh);

	Image GetAmplitudeImage();
};