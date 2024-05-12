#include<cmath>
#include"Fourier.h"

namespace Fourier {
	static const double PI = acos(-1.0);

	// ��λ��ת��N: 2���ݣ�ֻ�÷�ת�� log N λ.
	int bit_reverse(int i, int N) {
		// ֻ�ÿ��ǵ�num_bits��������λ.
		int num_bits;
		for (num_bits = 1; ((N >> num_bits) & 1) == 0; num_bits++);
		int ret = 0;
		for (int j = 0; j < num_bits; ++j) {
			ret = ret | ((1 & (i >> j)) << (num_bits - j - 1));
		}
		return ret;
	}

	void bit_reverse_displacement(ComplexDouble* target, int N, ComplexDouble* source, int length) {
		// ���target �� source��ͬһ���������´���һ����ʱ��
		std::shared_ptr<ComplexDouble[]> tmp;
		if (target == source) {
			// ����һ����ʱbuffer, ��source���ƹ���.
			tmp = std::shared_ptr<ComplexDouble[]>(new ComplexDouble[length]());
			std::copy(source, source + length, tmp.get());
			source = tmp.get();
		}
		for (int i = 0; i < N; ++i) {
			// λ��iԭ��Ӧ�����ĸ��±��������i��λ��ת����
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
			if (((i << j) & 0x80000000) != 0) {  // ���λ1������ 31-i λ����0��ʼ�ǣ�
				n = 1 << (32 - j);
				break;
			}
		}
		return n;
	}

	void fft1d(ComplexDouble* target, int N, ComplexDouble* data, int length, int inv) {
		if ((N & (N - 1)) != 0) {
			throw "N������2����������";
		}
		bit_reverse_displacement(target, N, data, length);
		
		// ��������任������ e^{-jk2pi/N}���������任������ e^{jk2pi/N}
		double m = inv ? 1 : -1;   // ���任
		// �Ե����ϼ��㣬�ӵڶ��㿪ʼ���ڶ���N=2����һ��N=1��,���ڶ��� w^k_2����G(w^k_1)��H(w^k_1)����
		for (int l = 2; l <= N; l *= 2) {
			// ��һ��k=0,1,2,...,N-1ʱ����Ҫ���� w^k_l���仯��λ�ǳ��� w^1_l = e^{-j2pi/l}����Ҫ�õ� w^k_{N/2}
			ComplexDouble w_unit(cos(2 * PI / l), sin(m * 2 * PI / l));
			// ��һ���� N / l �����㵥Ԫ��ÿ�����㵥Ԫ��Ӧ��range����ʼλ����j, ÿ��Ҫ���l��Ԫ��ȥ����һ�����㵥Ԫ�Ŀ�ͷ.
			for (int j = 0; j < N; j += l) {
				ComplexDouble wkl(1, 0);   // ��һ��Ԫ�ص�ʱ��k=0, ֱ����1.
				// ����Ƶ��k=0,1,2,...
				for (int k = 0; k < l / 2; ++k) {
					// ���� f(w^k_l) = G(w^k_{l/2}) + w^k_l * H(w^k_{l/2}), l/2ָ�ľ��ǡ���һ�㡱
					// f(w^{k+l/2}_l) = = G(w^k_{l/2}) - w^k_l * H(w^k_{l/2})
					ComplexDouble g = target[j + k];
					ComplexDouble h = wkl * target[j + k + l / 2];
					target[j + k] = g + h;
					target[j + k + l / 2] = g - h;
					wkl = wkl * w_unit;
				}
			}
		}

		// �������任���ͻ�Ҫ���� N
		if (inv) {
			for (int i = 0; i < N; ++i) {
				target[i] /= N;
			}
		}
	}

	std::shared_ptr<ComplexDouble[]> fft1d(ComplexDouble* data, int length, int& N, int inv) {
		// λ����任����������.
		// ���Ȳ���Ϊ��ӽ���2����.
		if (length < 2) {
			throw "���ڸ���Ҷ�任�����鳤�Ȳ���Ϊ0��1����Ϊ����";
		}
		int n;
		if (inv) {
			n = N;
		}
		else {
			N = n = find_nearest_larger_exp2(length);
		}

		// ����һ���ڴ����ڱ��渵��Ҷ�任���
		std::shared_ptr<ComplexDouble[]> fourier(new ComplexDouble[n]);

		fft1d(fourier.get(), n, data, length, inv);

		return fourier;
	}

	std::shared_ptr<ComplexDouble[]> fft1d(std::shared_ptr<ComplexDouble[]> data, int length, int& N, int inv) {
		return fft1d(data.get(), length, N, inv);
	}

	std::shared_ptr<ComplexDouble[]> fft2d(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N, int inv) {
		// ���ζ������С�������ִ��һά����Ҷ�任����
		if (W < 2 || H < 2)
			throw "��߶���ӦС��2.";

		ComplexDouble* headptr = data.get();
		int m, n;
		if (inv) {
			m = M, n = N;
		}
		else {
			M = m = find_nearest_larger_exp2(W);
			N = n = find_nearest_larger_exp2(H);
		} 
		// M(m)�У�N(n)��. ÿ��m��Ԫ�أ�ÿ��n��Ԫ��

		// �������������任������һ���ڴ��������.
		std::shared_ptr<ComplexDouble[]> fourier2d(new ComplexDouble[m * n]());  // ȫ����ʼ��Ϊ0.
		// �����ζ�ÿ����һά����Ҷ�任, ע��ԭͼֻ��H��. fourier2d��ʣ�µ���ȫ��0.
		for (int i = 0; i < H; ++i) {
			fft1d(fourier2d.get() + i * m, M, data.get() + i * W, W, inv);
		}
		// Ȼ���ڵ��µ�fourier2d�Ļ����ϣ���ÿ����һά����Ҷ�任��ע��Ҫ�����ڴ������Ե�����.
		std::shared_ptr<ComplexDouble[]> tmpbuf_src(new ComplexDouble[n]);
		std::shared_ptr<ComplexDouble[]> tmpbuf_dst(new ComplexDouble[n]);   // ����Ӧ�ÿ���ֱ�ӹ��ã�֮������
		for (int i = 0; i < m; ++i) {
			// �Ȱ�ͼƬfourier2d���е����ݸ��ƹ���.
			for (int j = 0; j < n; ++j) {
				int ind = i + j * m;
				tmpbuf_src[j] = fourier2d[ind];
			}
			// �����н���1d����Ҷ�任������ŵ�tmpbuf_dst.
			fft1d(tmpbuf_dst.get(), n, tmpbuf_src.get(), n, inv);
			// �ѽ���ŵ����������.
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

	// dst��Ŀ���ڴ棬����ΪN��data_extended�����0��չ��ģ�����Ϊ2N. data_extended�ᱻ����.
	void dct1d(ComplexDouble* dst, ComplexDouble* data_extended, int N, int inv) {
		//fft1d(data_extended, 2 * N, data_extended, 2 * N, inv);
		// DCT: 2cos(pi*k/(2N))Re(FFT) + 2sin(pi*k/(2N))Lm(FFT)
		// IDCT: ����fft..., ��ͬ��DCT-III / 2N.
		if (!inv) {   // ���任.
			fft1d(data_extended, 2 * N, data_extended, 2 * N, 0);
			for (int k = 0; k < N; ++k) {
				double t = PI * k / 2 / N;
				dst[k] = data_extended[k].real() * 2 * cos(t) + data_extended[k].imag() * 2 * sin(t);
			}
		}
		else {   // ��任.
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
		// �������ᵽ2���ݣ�Ȼ�󳤶ȷ�������Ĳ�0��ֱ����fft�������������任��� alpha, �������任����� alpha �ٳ��� M or N.
		int n = inv ? N : find_nearest_larger_exp2(length);
		int exten = 2 * n;
		N = n;
		// �������أ��ӳ�����. ע���N��Ҫȡ����
		std::shared_ptr<ComplexDouble[]> buf(new ComplexDouble[2 * n]());
		for (int i = 0; i < length; ++i) {
			buf[i] = data[i];
		}

		// ���ֻȡǰN.
		std::shared_ptr<ComplexDouble[]> ret(new ComplexDouble[n]());

		dct1d(ret.get(), buf.get(), n, inv);

		return ret;
	}

	std::shared_ptr<ComplexDouble[]> dct2d(std::shared_ptr<ComplexDouble[]> data, int W, int H, int& M, int& N, int inv) {
		int n = inv ? N : find_nearest_larger_exp2(H), m = inv ? M : find_nearest_larger_exp2(W);
		int n_exten = 2 * n, m_exten = 2 * m;
		M = m, N = n;

		// �������
		std::shared_ptr<ComplexDouble[]> ret(new ComplexDouble[m * n]());
		// ��ʱ������еľ��������Ű������ơ����ء�������ע�ⳤ��������
		std::shared_ptr<ComplexDouble[]> buf(new ComplexDouble[2 * (m > n ? m : n)]());
		// �ȱ任ÿһ��. ԭ������ֻ��H�У�ÿ��ֻ��W��Ԫ��
		for (int i = 0; i < H; ++i) {
			// ���и���. ע�����ı��벹Ϊ0.
			for (int j = 0; j < W; ++j) {
				buf[j] = data[i * W + j];
			}
			for (int j = W; j < 2 * m; ++j) {
				buf[j] = 0;
			}
			// ����һάDCT, ��Ϊret�����������ģ�����ֱ�Ӱѽ���Ž�ret
			//dct1d(buf.get(), buf.get(), m, inv);
			dct1d(ret.get() + i * M, buf.get(), m, inv);
		}
		// ��ÿ�н��б任��Ҫ��ret��������ԭ����.
		for (int i = 0; i < m; ++i) {
			// ���и���, ��ı��벹��Ϊ0
			for (int j = 0; j < n; ++j) {
				buf[j] = ret[j * m + i];
			}
			for (int j = n; j < 2 * n; ++j) {
				buf[j] = 0;
			}
			// ����һάDCT���в����������Բ���ֱ�ӷŽ�ret
			dct1d(buf.get(), buf.get(), n, inv);
			// ���ƽ�ret.
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