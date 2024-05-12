// DigitalImageHW2.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。

#include<cstdio>
#include <iostream>
#include<memory>
#include "Image.h"
#include"Spectrum.h"
#include"Fourier.h"

// 如果H不为0，表示是二维，否则是1维
void PRINT_COMPLEXDOUBLE_ARRAY(const std::shared_ptr<ComplexDouble[]> data, int W, int H) {
    if (H == 0) {
        for (int i = 0; i < W; ++i) {
            printf("%lf + %lf j\n", data[i].real(), data[i].imag());
        }
    }
    else {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                int off = i * W + j;
                printf("%lf + %lf j, ", data[off].real(), data[off].imag());
            }
            printf("\n");
        }
    }
}

int main()
{
    /*string fnames[] = {"image/Baboon", "image/dance_crop_gray", "image/Cameraman", "image/Goldhill"};
    int len = sizeof(fnames) / sizeof(string);
    for (int i = 0; i < len; ++i) {
        Image img(fnames[i] + ".jpg");
        img = img.ConvertToGray();
        Spectrum spec_fft(img, Fourier::image_fft, 1);
        Spectrum spec_dct(img, Fourier::image_dct, 1);
        Image inv_fft = spec_fft.InverseTransform(Fourier::image_inv_fft);
        Image inv_dct = spec_dct.InverseTransform(Fourier::image_inv_dct);
        inv_fft.Write(fnames[i] + "_fft_inv.jpg");
        inv_dct.Write(fnames[i] + "_dct_inv.jpg");
        printf((fnames[i] + ": done").c_str());
    }*/

    //Image amplitude_image = spec.GetAmplitudeImage();
    //amplitude_image.Write(fname + "_fft.jpg");
    //amplitude_image.Write(fname + "_dct.jpg");

    string fname = "image/Goldhill";
    Image img(fname + ".jpg");
    img = img.ConvertToGray();
    Spectrum spec(img, Fourier::image_fft, 1);
    spec.BandPassFilter(10, 50);
    Image amplitude = spec.GetAmplitudeImage();
    amplitude.Write(fname + "_fft_amp_bandpass.jpg");
    Image inv = spec.InverseTransform(Fourier::image_inv_fft);
    inv.Write(fname + "_fft_bandpass.jpg");

    return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
