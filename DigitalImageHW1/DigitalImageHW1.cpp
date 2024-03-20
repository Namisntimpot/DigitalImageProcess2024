// DigitalImageHW1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// 避免重复引入 stbi_xxx.h
#include "Image.h"

int main()
{
    const int n_imgs = 2;
    const string imgs[] = {"image/sleepcat","image/yunnan"};
    for (int i = 0; i < n_imgs; ++i) {
        const string origin = imgs[i] + ".jpg";
        Image im(origin);
        Histogram hist(im);
        hist.Draw(imgs[i] + "_hist.png");
        hist.Equalize();
        Image equalized = im.PointTransformFromTable(hist.GetLevelMapTable());
        equalized.Write(imgs[i] + "_equalized.jpg");
        Histogram equalized_hist(equalized);
        equalized_hist.Draw(imgs[i] + "_hist_equalized.png");

        // 线性拉伸
        Image transformed1 = im.PointTransformLinear(50, 200, 120, 160);
        transformed1.Write(imgs[i] + "_linear1.jpg");
        Image transformed2 = im.PointTransformLinear(100, 150, 20, 255);
        transformed2.Write(imgs[i] + "_linear2.jpg");
    }
    //const char* p_cat = "image/sleepcat.jpg";
    //const char* p_yunnan = "image/yunnan.jpg";
    //const char* p = "image/sleepcat.jpg";
    //Image img(p);
    //Histogram hist(img);
    //hist.Draw("image/sleepcat_hist.png");
    //hist.Equalize();
    //Image equalized = img.PointTransformFromTable(hist.GetLevelMapTable());
    //Histogram equalized_hist(equalized);
    //equalized_hist.Draw("image/sleepcat_equalized_hist.png");
    //string output("image/sleepcat_histeq.jpg");
    //equalized.Write(output);
    //Image linear = img.PointTransformLinear(0, 255, 100, 150);
    //string output("image/sleepcat_linear_transform.jpg");
    //linear.Write(output);
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
