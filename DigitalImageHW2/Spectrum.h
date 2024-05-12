#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// �����ظ�����stb_image, ֻ��main.cpp������һ�μ���.

#include<algorithm>
#include<memory>
#include<complex>
#include<functional>

#include"Image.h"

typedef std::complex<double> ComplexDouble;
// ���任����ָ��, W, H, M, N, �������������.
typedef std::function<std::shared_ptr<ComplexDouble[]>(std::shared_ptr<ComplexDouble[]>, int, int, int&, int&)> SpectrumTransformFunc;  
// ��任����ָ��, ��һ��int��width, �ڶ���int��height
typedef std::function<std::shared_ptr<ComplexDouble[]>(std::shared_ptr<ComplexDouble[]>, int, int)> SpectrumInverseTransformFunc;
typedef std::function<ComplexDouble(ComplexDouble, int, int, int, int)> SpectrumFilterFunc;  // �˲�������ָ��,   ��һ��int�Ǻ����u���ڶ���int�������v, ������int�Ǻ������ģ����ĸ�int����������

class Spectrum {
public:
	int M, N, H, W, shifted;   // M ��Ӧx���Ǻ��᣻N ��Ӧy��������.  H, W ��ԭͼ�Ŀ��
	std::shared_ptr<ComplexDouble[]> rawdata;

	// ��ͼƬת����complex���������Ļ������⣬����func���б任.
	Spectrum(Image img, SpectrumTransformFunc transform_func, int shift);
	// ��Ƶ�״���func������任���������Ļ�����, ���u8 Image.
	Image InverseTransform(SpectrumInverseTransformFunc inversefunc);
	void ApplyFilter(SpectrumFilterFunc filterfunc);         // ֱ���ڱ������Ϸ����޸�.
	void LowPassFilter(int thresh);
	void BandPassFilter(int l_thresh, int h_thresh);
	void HighPassFilter(int thresh);

	Image GetAmplitudeImage();
};