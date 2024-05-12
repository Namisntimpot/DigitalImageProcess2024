#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// 避免重复引入stb_image, 只在main.cpp中引入一次即可.

#include<string>
#include<memory>
#include<algorithm>
#include<complex>

#include "Image.h"
#include "Spectrum.h"

namespace Fourier {
	// 如果是正变换，则W,H是输入、M,N是输出；如果是逆变换，则四个都是输入.
	// 如果是正变换，则W,H是data宽高，MN是上取到最近的2的幂之后的大小；如果是逆变换，则MN仍旧是上取到最近的2的幂之后的大小，WH是变换后图片需要的宽高.
	std::shared_ptr<ComplexDouble[]> fft2d(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N, int inv);
	// 进行一维傅里叶变换，把结果放到target中，N必须严格是2的幂.
	void fft1d(ComplexDouble* target, int N, ComplexDouble* data, int length, int inv);
	// 实现蝶形算法，如果是正变换，则N是输出. 逆变换则都是输入
	// 如果是正变换，length是数组长度，N是填充为最近2的幂之后的长度（频谱长度）如果是逆变换，则N是频谱长度，length是逆变换后需要的长度.
	std::shared_ptr<ComplexDouble[]> fft1d(std::shared_ptr<ComplexDouble[]> data, int length, int& N, int inv);  
	// 实现鲽形算法，如果是正变换，则N是输出. 逆变换则都是输入
	// 如果是正变换，length是数组长度，N是填充为最近2的幂之后的长度（频谱长度）如果是逆变换，则N是频谱长度，length是逆变换后需要的长度.
	std::shared_ptr<ComplexDouble[]> fft1d(ComplexDouble* data, int length, int& N, int inv);

	// M,N是输出，shift表示频谱中心化. M, N是输出，表示将宽高填充到2的幂.
	//std::shared_ptr<ComplexDouble[]> image_fft(const Image img, int& M, int& N, int shift = 1);
	std::shared_ptr<ComplexDouble[]> image_fft(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N);
	// W,H表示图像宽高, 因为频谱是填充过0的.
	//Image image_inv_fft(Spectrum spectrum, int W, int H);
	std::shared_ptr<ComplexDouble[]> image_inv_fft(std::shared_ptr<ComplexDouble[]> spec, int M, int N);

	// 如果是正变换，length是数组长度，N是填充为最近2的幂之后的长度（频谱长度）如果是逆变换，则N是频谱长度，length是逆变换后需要的长度.
	std::shared_ptr<ComplexDouble[]> dct1d(ComplexDouble* data, int length, int& N, int inv);

	std::shared_ptr<ComplexDouble[]> dct2d(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N, int inv);

	std::shared_ptr<ComplexDouble[]> image_dct(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N);

	std::shared_ptr<ComplexDouble[]> image_inv_dct(std::shared_ptr<ComplexDouble[]> spec, int M, int N);
}