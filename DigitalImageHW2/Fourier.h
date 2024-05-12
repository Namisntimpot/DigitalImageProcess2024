#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// �����ظ�����stb_image, ֻ��main.cpp������һ�μ���.

#include<string>
#include<memory>
#include<algorithm>
#include<complex>

#include "Image.h"
#include "Spectrum.h"

namespace Fourier {
	// ��������任����W,H�����롢M,N��������������任�����ĸ���������.
	// ��������任����W,H��data��ߣ�MN����ȡ�������2����֮��Ĵ�С���������任����MN�Ծ�����ȡ�������2����֮��Ĵ�С��WH�Ǳ任��ͼƬ��Ҫ�Ŀ��.
	std::shared_ptr<ComplexDouble[]> fft2d(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N, int inv);
	// ����һά����Ҷ�任���ѽ���ŵ�target�У�N�����ϸ���2����.
	void fft1d(ComplexDouble* target, int N, ComplexDouble* data, int length, int inv);
	// ʵ�ֵ����㷨����������任����N�����. ��任��������
	// ��������任��length�����鳤�ȣ�N�����Ϊ���2����֮��ĳ��ȣ�Ƶ�׳��ȣ��������任����N��Ƶ�׳��ȣ�length����任����Ҫ�ĳ���.
	std::shared_ptr<ComplexDouble[]> fft1d(std::shared_ptr<ComplexDouble[]> data, int length, int& N, int inv);  
	// ʵ�������㷨����������任����N�����. ��任��������
	// ��������任��length�����鳤�ȣ�N�����Ϊ���2����֮��ĳ��ȣ�Ƶ�׳��ȣ��������任����N��Ƶ�׳��ȣ�length����任����Ҫ�ĳ���.
	std::shared_ptr<ComplexDouble[]> fft1d(ComplexDouble* data, int length, int& N, int inv);

	// M,N�������shift��ʾƵ�����Ļ�. M, N���������ʾ�������䵽2����.
	//std::shared_ptr<ComplexDouble[]> image_fft(const Image img, int& M, int& N, int shift = 1);
	std::shared_ptr<ComplexDouble[]> image_fft(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N);
	// W,H��ʾͼ����, ��ΪƵ��������0��.
	//Image image_inv_fft(Spectrum spectrum, int W, int H);
	std::shared_ptr<ComplexDouble[]> image_inv_fft(std::shared_ptr<ComplexDouble[]> spec, int M, int N);

	// ��������任��length�����鳤�ȣ�N�����Ϊ���2����֮��ĳ��ȣ�Ƶ�׳��ȣ��������任����N��Ƶ�׳��ȣ�length����任����Ҫ�ĳ���.
	std::shared_ptr<ComplexDouble[]> dct1d(ComplexDouble* data, int length, int& N, int inv);

	std::shared_ptr<ComplexDouble[]> dct2d(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N, int inv);

	std::shared_ptr<ComplexDouble[]> image_dct(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N);

	std::shared_ptr<ComplexDouble[]> image_inv_dct(std::shared_ptr<ComplexDouble[]> spec, int M, int N);
}