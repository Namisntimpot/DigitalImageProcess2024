#include "Spectrum.h"

Spectrum::Spectrum(Image img, SpectrumTransformFunc transform_func, int shift) : H(img.height), W(img.width), shifted(shift) {
	if (img.channels != 1)
		throw "目前只支持灰度图片，通道数应为1";

	// 把uchar的图片数值复制到复数数组中去
	std::shared_ptr<ComplexDouble[]> src(new ComplexDouble[img.height * img.width]);  
	const std::shared_ptr<unsigned char[]> img_u8_ptr = img.GetRawDataPtr();
	std::transform(img_u8_ptr.get(), img_u8_ptr.get() + img.height * img.width, src.get(),
		[](unsigned char v) -> ComplexDouble {
			return ComplexDouble(static_cast<double>(v), 0.0);
		});

	// 如果要执行频谱中心化，需要对每个像素乘以 (-1)^(x+y)
	if (shift) {
		for (int i = 0; i < img.height; ++i) {
			for (int j = 0; j < img.width; ++j) {
				int mul = ((i + j) & 1) ? -1 : 1;
				int ind = i * img.width + j;
				src[ind] *= mul;
			}
		}
	}

	// 执行变换.
	M = N = 0;
	rawdata = transform_func(src, W, H, M, N);
}

Image Spectrum::InverseTransform(SpectrumInverseTransformFunc inversefunc) {
	// 执行逆变换
	auto ret = inversefunc(rawdata, M, N);

	// 处理中心化问题, 就是直接变成正数即可.
	Image img(H, W, 1);
	std::shared_ptr<unsigned char[]> img_raw_data = img.GetRawDataPtr();
	for (int i = 0; i < H; ++i) {
		for (int j = 0; j < W; ++j) {
			int complex_ind = i * M + j;
			int uchar_ind = i * W + j;
			img_raw_data[uchar_ind] = static_cast<unsigned char>(abs(ret[complex_ind].real()));
		}
	}

	return img;
}

Image Spectrum::GetAmplitudeImage() {
	// 1. 计算幅度（abs）
	// 2. 对数变换进行scale，(log(1+amplitude)
	// 3. 归一化：减去min，除以max-min
	// 4. 乘以255，输出图片.
	Image amplitude(N, M, 1);
	std::shared_ptr<double[]> tmp(new double[N * M]());
	double min_amp = 1e10, max_amp = -1;
	// 幅度，log.
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < M; ++j) {
			int off = i * M + j;
			double v = log(1 + abs(rawdata[off]));
			tmp[off] = v;
			min_amp = min_amp > v ? v : min_amp;
			max_amp = max_amp < v ? v : max_amp;
		}
	}
	// 归一化, 0-255
	double scale = max_amp - min_amp;
	auto imgdata = amplitude.GetRawDataPtr();
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < M; ++j) {
			int off = i * M + j;
			double v = (tmp[off] - min_amp) / scale * MAX_COLOR_LEVEL_VALUE;
			imgdata[off] = static_cast<unsigned char>(v);
		}
	}
	return amplitude;
}

void Spectrum::ApplyFilter(SpectrumFilterFunc filterfunc) {         // 直接在本对象上发生修改.
	int mid_M = M / 2;
	int mid_N = N / 2;
	for (int v = 0; v < N; ++v) {
		for (int u = 0; u < M; ++u) {
			int off = v * M + u;
			rawdata[off] = filterfunc(rawdata[off], u, v, mid_M, mid_N);
		}
	}
}

// 低通滤波，直接把高频的过滤掉！低频的不变
void Spectrum::LowPassFilter(int thresh) {
	int thresh2 = thresh * thresh;
	auto filter = [=](ComplexDouble data, int u, int v, int mid_M, int mid_N)->ComplexDouble {
		return (u - mid_M) * (u - mid_M) + (v - mid_N) * (v - mid_N) > thresh2 ? 0 : data;
		};
	ApplyFilter(filter);
}

// 带通滤波，把中间频率的留下，两侧的滤掉
void Spectrum::BandPassFilter(int l_thresh, int h_thresh) {
	int l_thresh2 = l_thresh * l_thresh, h_thresh2 = h_thresh * h_thresh;
	auto filter = [=](ComplexDouble data, int u, int v, int mid_M, int mid_N) -> ComplexDouble {
		int r2 = (u - mid_M) * (u - mid_M) + (v - mid_N) * (v - mid_N);
		return r2 > l_thresh2 && r2 < h_thresh2 ? data : 0;
	};
	ApplyFilter(filter);
}

// 高通滤波，直接把低频的过滤掉，高频的不变
void Spectrum::HighPassFilter(int thresh) {
	int thresh2 = thresh * thresh;
	auto filter = [=](ComplexDouble data, int u, int v, int mid_M, int mid_N) -> ComplexDouble {
		int r2 = (u - mid_M) * (u - mid_M) + (v - mid_N) * (v - mid_N);
		return r2 > thresh2 ? data : 0;
	};
	ApplyFilter(filter);
}