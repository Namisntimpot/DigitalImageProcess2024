#include<cmath>
#include"Fourier.h"

namespace Fourier {
	static const double PI = acos(-1.0);

	// 按位翻转；N: 2的幂，只用翻转低 log N 位.
	int bit_reverse(int i, int N) {
		// 只用考虑低num_bits个二进制位.
		int num_bits;
		for (num_bits = 1; ((N >> num_bits) & 1) == 0; num_bits++);
		int ret = 0;
		for (int j = 0; j < num_bits; ++j) {
			ret = ret | ((1 & (i >> j)) << (num_bits - j - 1));
		}
		return ret;
	}

	void bit_reverse_displacement(ComplexDouble* target, int N, ComplexDouble* source, int length) {
		// 如果target 和 source是同一个，必须新创建一个暂时的
		std::shared_ptr<ComplexDouble[]> tmp;
		if (target == source) {
			// 创建一个临时buffer, 把source复制过来.
			tmp = std::shared_ptr<ComplexDouble[]>(new ComplexDouble[length]());
			std::copy(source, source + length, tmp.get());
			source = tmp.get();
		}
		for (int i = 0; i < N; ++i) {
			// 位置i原本应该是哪个下标的数，将i按位翻转即可
			int rev = bit_reverse(i, N);
			if (rev >= length) {
				target[i] = 0;
			}
			else {
				target[i] = source[rev];
			}
		}
	}

	int find_nearest_larger_exp2(int i) {
		if ((i & (i - 1)) == 0)
			return i;
		int n;
		for (int j = 0; j < 32; ++j) {
			if (((i << j) & 0x80000000) != 0) {  // 最高位1出现在 31-i 位（从0开始记）
				n = 1 << (32 - j);
				break;
			}
		}
		return n;
	}

	void fft1d(ComplexDouble* target, int N, ComplexDouble* data, int length, int inv) {
		if ((N & (N - 1)) != 0) {
			throw "N必须是2的正整数幂";
		}
		bit_reverse_displacement(target, N, data, length);
		
		// 如果是正变换，带入 e^{-jk2pi/N}，如果是逆变换，带入 e^{jk2pi/N}
		double m = inv ? 1 : -1;   // 正变换
		// 自底向上计算，从第二层开始，第二层N=2（第一层N=1）,即第二层 w^k_2，用G(w^k_1)和H(w^k_1)来算
		for (int l = 2; l <= N; l *= 2) {
			// 这一层k=0,1,2,...,N-1时，需要计算 w^k_l，变化单位是乘以 w^1_l = e^{-j2pi/l}。需要用到 w^k_{N/2}
			ComplexDouble w_unit(cos(2 * PI / l), sin(m * 2 * PI / l));
			// 这一层有 N / l 个运算单元，每个运算单元对应的range的起始位置是j, 每次要跨过l个元素去到下一个运算单元的开头.
			for (int j = 0; j < N; j += l) {
				ComplexDouble wkl(1, 0);   // 第一个元素的时候k=0, 直接是1.
				// 计算频率k=0,1,2,...
				for (int k = 0; k < l / 2; ++k) {
					// 计算 f(w^k_l) = G(w^k_{l/2}) + w^k_l * H(w^k_{l/2}), l/2指的就是“上一层”
					// f(w^{k+l/2}_l) = = G(w^k_{l/2}) - w^k_l * H(w^k_{l/2})
					ComplexDouble g = target[j + k];
					ComplexDouble h = wkl * target[j + k + l / 2];
					target[j + k] = g + h;
					target[j + k + l / 2] = g - h;
					wkl = wkl * w_unit;
				}
			}
		}

		// 如果是逆变换，就还要除以 N
		if (inv) {
			for (int i = 0; i < N; ++i) {
				target[i] /= N;
			}
		}
	}

	std::shared_ptr<ComplexDouble[]> fft1d(ComplexDouble* data, int length, int& N, int inv) {
		// 位逆序变换，重新排列.
		// 首先补充为最接近的2的幂.
		if (length < 2) {
			throw "用于傅里叶变换的数组长度不能为0、1或者为负数";
		}
		int n;
		if (inv) {
			n = N;
		}
		else {
			N = n = find_nearest_larger_exp2(length);
		}

		// 申请一块内存用于保存傅里叶变换结果
		std::shared_ptr<ComplexDouble[]> fourier(new ComplexDouble[n]);

		fft1d(fourier.get(), n, data, length, inv);

		return fourier;
	}

	std::shared_ptr<ComplexDouble[]> fft1d(std::shared_ptr<ComplexDouble[]> data, int length, int& N, int inv) {
		return fft1d(data.get(), length, N, inv);
	}

	std::shared_ptr<ComplexDouble[]> fft2d(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N, int inv) {
		// 依次对所有行、所有列执行一维傅里叶变换即可
		if (W < 2 || H < 2)
			throw "宽高都不应小于2.";

		ComplexDouble* headptr = data.get();
		int m, n;
		if (inv) {
			m = M, n = N;
		}
		else {
			M = m = find_nearest_larger_exp2(W);
			N = n = find_nearest_larger_exp2(H);
		} 
		// M(m)列，N(n)行. 每行m个元素，每列n个元素

		// 对所有行列做变换，处理一下内存管理问题.
		std::shared_ptr<ComplexDouble[]> fourier2d(new ComplexDouble[m * n]());  // 全部初始化为0.
		// 先依次对每行做一维傅里叶变换, 注意原图只有H行. fourier2d中剩下的行全是0.
		for (int i = 0; i < H; ++i) {
			fft1d(fourier2d.get() + i * m, M, data.get() + i * W, W, inv);
		}
		// 然后在当下的fourier2d的基础上，对每列做一维傅里叶变换，注意要处理内存连续性的问题.
		std::shared_ptr<ComplexDouble[]> tmpbuf_src(new ComplexDouble[n]);
		std::shared_ptr<ComplexDouble[]> tmpbuf_dst(new ComplexDouble[n]);   // 这俩应该可以直接共用，之后试试
		for (int i = 0; i < m; ++i) {
			// 先把图片fourier2d中列的内容复制过来.
			for (int j = 0; j < n; ++j) {
				int ind = i + j * m;
				tmpbuf_src[j] = fourier2d[ind];
			}
			// 对这列进行1d傅里叶变换，结果放到tmpbuf_dst.
			fft1d(tmpbuf_dst.get(), n, tmpbuf_src.get(), n, inv);
			// 把结果放到矩阵的列中.
			for (int j = 0; j < n; ++j) {
				int ind = i + j * m;
				fourier2d[ind] = tmpbuf_dst[j];
			}
		}

		return fourier2d;
	}

	std::shared_ptr<ComplexDouble[]> image_fft(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N) {
		auto ret = fft2d(data, W, H, M, N, 0);
		return ret;
	}

	std::shared_ptr<ComplexDouble[]> image_inv_fft(std::shared_ptr<ComplexDouble[]> spec, int M, int N) {
		int m = M, n = N;
		auto ret = fft2d(spec, M, N, m, n, 1);
		return ret;
	}

	//---------DCT--------------

	double dct_alpha(int i, int N) {
		double mul = i == 0 ? 1 : 2;
		return sqrt(mul / N);
	}

	// dst是目标内存，长度为N，data_extended是填充0扩展后的，长度为2N. data_extended会被覆盖.
	void dct1d(ComplexDouble* dst, ComplexDouble* data_extended, int N, int inv) {
		//fft1d(data_extended, 2 * N, data_extended, 2 * N, inv);
		// DCT: 2cos(pi*k/(2N))Re(FFT) + 2sin(pi*k/(2N))Lm(FFT)
		// IDCT: 两次fft..., 等同于DCT-III / 2N.
		if (!inv) {   // 正变换.
			fft1d(data_extended, 2 * N, data_extended, 2 * N, 0);
			for (int k = 0; k < N; ++k) {
				double t = PI * k / 2 / N;
				dst[k] = data_extended[k].real() * 2 * cos(t) + data_extended[k].imag() * 2 * sin(t);
			}
		}
		else {   // 逆变换.
			double x0 = data_extended[0].real();
			std::shared_ptr<ComplexDouble[]> sinbuf(new ComplexDouble[2 * N]());
			for (int n = 0; n < N; ++n) {
				double t = PI * n / 2 / N;
				ComplexDouble origin = data_extended[n];
				data_extended[n] = origin * cos(t);
				sinbuf[n] = origin * sin(t);
			}
			fft1d(data_extended, 2 * N, data_extended, 2 * N, 0);
			fft1d(sinbuf.get(), 2 * N, sinbuf.get(), 2 * N, 0);
			for (int n = 0; n < N; ++n) {
				dst[n] = (2 * data_extended[n].real() + 2 * sinbuf[n].imag() - x0) / N / 2;
			}
		}
	}

	std::shared_ptr<ComplexDouble[]> dct1d(ComplexDouble* data, int length, int& N, int inv) {
		// F(u,v) = alpha(u, M) alpha(v, N) \sum...
		// alpha(u, M) = \sqrt{1 / M} if u == 0 else \sqrt{2 / M}
		// 把数组提到2的幂，然后长度翻倍，多的补0，直接用fft，最后，如果是正变换则乘 alpha, 如果是逆变换则乘以 alpha 再乘以 M or N.
		int n = inv ? N : find_nearest_larger_exp2(length);
		int exten = 2 * n;
		N = n;
		// 进行延拓，延长两倍. 注意后N个要取反序
		std::shared_ptr<ComplexDouble[]> buf(new ComplexDouble[2 * n]());
		for (int i = 0; i < length; ++i) {
			buf[i] = data[i];
		}

		// 结果只取前N.
		std::shared_ptr<ComplexDouble[]> ret(new ComplexDouble[n]());

		dct1d(ret.get(), buf.get(), n, inv);

		return ret;
	}

	std::shared_ptr<ComplexDouble[]> dct2d(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N, int inv) {
		int n = inv ? N : find_nearest_larger_exp2(H), m = inv ? M : find_nearest_larger_exp2(W);
		int n_exten = 2 * n, m_exten = 2 * m;
		M = m, N = n;

		// 结果矩阵
		std::shared_ptr<ComplexDouble[]> ret(new ComplexDouble[m * n]());
		// 临时存放行列的矩阵，这个存放包括复制、延拓、倍增，注意长度是两倍
		std::shared_ptr<ComplexDouble[]> buf(new ComplexDouble[2 * (m > n ? m : n)]());
		// 先变换每一行. 原数据中只有H行，每行只有W个元素
		for (int i = 0; i < H; ++i) {
			// 进行复制. 注意多余的必须补为0.
			for (int j = 0; j < W; ++j) {
				buf[j] = data[i * W + j];
			}
			for (int j = W; j < 2 * m; ++j) {
				buf[j] = 0;
			}
			// 进行一维DCT, 因为ret中行是连续的，可以直接把结果放进ret
			//dct1d(buf.get(), buf.get(), m, inv);
			dct1d(ret.get() + i * M, buf.get(), m, inv);
		}
		// 对每列进行变换，要用ret，而不是原数据.
		for (int i = 0; i < m; ++i) {
			// 进行复制, 多的必须补充为0
			for (int j = 0; j < n; ++j) {
				buf[j] = ret[j * m + i];
			}
			for (int j = n; j < 2 * n; ++j) {
				buf[j] = 0;
			}
			// 进行一维DCT，列不连续，所以不能直接放进ret
			dct1d(buf.get(), buf.get(), n, inv);
			// 复制进ret.
			for (int j = 0; j < n; ++j) {
				ret[j * m + i] = buf[j];
			}
		}

		return ret;
	}

	std::shared_ptr<ComplexDouble[]> image_dct(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N) {
		auto ret = dct2d(data, W, H, M, N, 0);
		return ret;
	}

	std::shared_ptr<ComplexDouble[]> image_inv_dct(std::shared_ptr<ComplexDouble[]> spec, int M, int N) {
		int m = M, n = N;
		auto ret = dct2d(spec, M, N, m, n, 1);
		return ret;
	}
};